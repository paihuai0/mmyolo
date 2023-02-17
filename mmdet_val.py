import argparse
import math
import torch
from mmengine.logging import print_log
import numpy as np
import torch.nn as nn
from terminaltables import AsciiTable
from mmcv.ops import nms
from mmengine import Config
from mmengine.fileio import load
from mmengine.utils import ProgressBar
from mmdet.evaluation import bbox_overlaps
from mmdet.evaluation import eval_map, bbox_area
from mmdet.utils import replace_cfg_vals, update_data_root
from mmengine.runner import Runner

pair = nn.PairwiseDistance(p=2)


def calculate_num_confusion_matrix(dataset,
                                   results,
                                   score_thr=0.3,
                                   nms_iou_thr=None,
                                   area_size=None,
                                   tp_iou_thr=0.5):

    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        num_analyze_per_img_dets(confusion_matrix, gts,
                                 res_bboxes, score_thr, tp_iou_thr,
                                 nms_iou_thr, area_size)
        prog_bar.update()
    return confusion_matrix


def calculate_dis_confusion_matrix(dataset,
                                   results,
                                   score_thr=0.3,
                                   tp_iou_thr=0.5):
    num_classes = len(dataset.metainfo['classes'])
    result_matrix = np.zeros(shape=[num_classes, 2])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        dis_analyze_per_img_dets(result_matrix, gts, res_bboxes,
                                 score_thr, tp_iou_thr, )
        prog_bar.update()
    return result_matrix


def process_result(result, dataset_name):
    det_results = []
    dataset_name = dataset_name
    for i in range(len(result)):
        tensor_list = []

        tensor_result = torch.cat((result[i]['pred_instances']['bboxes'],
                                   torch.unsqueeze(result[i]['pred_instances']['scores'], dim=1)), dim=1)
        tensor_result = torch.cat((tensor_result, torch.unsqueeze(result[i]['pred_instances']['labels'], dim=1)), dim=1)
        tensor_list.append(tensor_result.numpy())
        labels = tensor_list[0][:, 5]
        temp_list = [[] for _ in range(len(dataset_name))]
        for i, label in enumerate(labels):
            temp_list[int(label)].append(tensor_list[0][i][:5])

        for i, data in enumerate(temp_list):
            if len(temp_list[i]) != 0:
                temp_list[i] = np.stack(data)
            else:
                temp_list[i] = np.array(temp_list[i]).reshape(0, 5)
        det_results.append(temp_list)

    return det_results


def voc_eval(result, dataset, iou_thr=0.5, nproc=1):
    annotations = [dataset.get_data_info(i) for i in range(len(dataset))]

    new_annotations_list = []
    for value in annotations:
        bboxes = []
        labels = []
        bboxes_ignore = []
        value_dict = value
        for i in range(len(value_dict['instances'])):
            bboxes.append(value_dict['instances'][i]['bbox'])
            labels.append(value_dict['instances'][i]['bbox_label'])
            bboxes_ignore.append(value_dict['instances'][i]['ignore_flag'])
        new_annotations_dict = {
            "bboxes": np.array(bboxes),
            "labels": np.array(labels),
            "bboxes_ignore": np.array(bboxes_ignore)
        }
        new_annotations_list.append(new_annotations_dict)

    dataset_name = dataset.metainfo['classes']
    det_results = process_result(result, dataset_name)
    _, eval_results = eval_map(
        det_results,
        new_annotations_list,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc)
    return eval_results


def dis_analyze_per_img_dets(
        result_matrix,
        gts,
        result,
        score_thr=0.3,
        tp_iou_thr=0.5,
):
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i,  score in enumerate(det_scores):
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr and gt_label == det_label:
                        det_point_y = det_bboxes[i][1] + (
                                (det_bboxes[i][3] - det_bboxes[i][1]) / 2)
                        det_point_x = det_bboxes[i][2]

                        gt_point_y = gt_bboxes[j][1] + (
                                (gt_bboxes[j][3] - gt_bboxes[j][1]) / 2)
                        gt_point_x = gt_bboxes[j][2]

                        point_val_result = math.sqrt(((gt_point_x - det_point_x) ** 2) +
                                                     ((gt_point_y - det_point_y) ** 2))
                        line_val_result = gt_bboxes[j][2] - det_bboxes[i][2]

                        result_matrix[gt_label, 0] += abs(point_val_result)
                        result_matrix[gt_label, 1] += abs(line_val_result)


def num_analyze_per_img_dets(confusion_matrix,
                             gts,
                             result,
                             score_thr=0.3,
                             tp_iou_thr=0.5,
                             nms_iou_thr=None,
                             area_size=None):

    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    if area_size:

        area_gt = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        mask = (area_gt > area_size[0]) & (area_gt <= area_size[1])
        gt_bboxes = gt_bboxes[mask]
        gt_labels = gt_labels[mask]

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask]
        det_scores = result['scores'][mask]

        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)

        ious = bbox_overlaps(det_bboxes[:, :4].numpy(), gt_bboxes)

        if area_size:
            mask = bbox_area(det_bboxes, area_size)
            det_scores = det_scores[mask]

        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='DT_detection test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl_path', help='pkl_path file')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--tp_iou_thr',
        type=float,
        default=0.5,
        help='iou threshold (default: 0.5)')
    parser.add_argument(
        '--area_size',
        type=float,
        default=None,
        help='test(val) bbox area range (default: None)')
    parser.add_argument(
        '--nms_iou_thr',
        type=float,
        default=None,
        help='nms iou  threshold (default: None)')

    args = parser.parse_args()

    return args


def main():
    # register_all_modules()
    exp = 1e-7
    args = parse_args()

    config = args.config

    pkl_path = args.pkl_path
    score_thr = args.score_thr
    tp_iou_thr = args.tp_iou_thr
    area_size = args.area_size
    nms_iou_thr = args.nms_iou_thr

    cfg = Config.fromfile(config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    runner = Runner.from_cfg(cfg)

    results = load(pkl_path)

    dataset = runner.test_dataloader.dataset
    dataset_name = dataset.metainfo['classes']

    # print('\n----------------datasets-------------------\n')
    # print(dataset)
    print('\n------------cal_val-----------------------\n')

    # 1.map
    eval_results = voc_eval(results, dataset, tp_iou_thr, 1)

    num_gts = np.zeros((1, len(dataset_name)), dtype=int)
    for i, cls_result in enumerate(eval_results):
        num_gts[:, i] = cls_result['num_gts']

    # 2. cal tp fp
    TP_confusion_matrix = calculate_num_confusion_matrix(
        dataset,
        results,
        score_thr=score_thr,
        nms_iou_thr=nms_iou_thr,
        tp_iou_thr=tp_iou_thr,
        area_size=area_size)
    np.set_printoptions(precision=4, suppress=True)
    tp = TP_confusion_matrix.diagonal()
    fp = TP_confusion_matrix.sum(0) - tp  # false positives
    pure_fp = TP_confusion_matrix[-1, :]
    confusion_fp = fp - pure_fp
    # fn = TP_confusion_matrix[:, -1]  # false negatives (missed detections)
    fn = num_gts[0] - tp[:-1]
    print('\n----------------cal_tp_fp_fn-------------------\n')

    header1 = ['class', 'tp', 'pure_fp', 'conf_fp', 'fn']
    table_data1 = [header1]
    for i in range(len(dataset_name)):
        row_data1 = [
            dataset_name[i], tp[i], pure_fp[i], confusion_fp[i], fn[i]
        ]
        table_data1.append(row_data1)
    table1 = AsciiTable(table_data1)
    print_log('\n' + table1.table)

    # 3. cal dis loss
    DIS_confusion_matrix = calculate_dis_confusion_matrix(
        dataset, results, score_thr=score_thr, tp_iou_thr=tp_iou_thr)
    np.set_printoptions(precision=4, suppress=True)

    print('\n-------------cal_point_line----------------------\n')
    point_result_normal = list(
        map(lambda x: x[0] / (x[1] + exp),
            zip(list(DIS_confusion_matrix[:, 0]), tp)))
    line_result_normal = list(
        map(lambda x: x[0] / (x[1] + exp),
            zip(list(DIS_confusion_matrix[:, 1]), tp)))

    header2 = ['class', 'point_result', 'line_result']
    table_data2 = [header2]
    for i in range(len(dataset_name)):
        row_data2 = [
            dataset_name[i], f'{point_result_normal[i]:.3f}', f'{line_result_normal[i]:.3f}'
        ]
        table_data2.append(row_data2)
    table2 = AsciiTable(table_data2)
    print_log('\n' + table2.table)


if __name__ == '__main__':
    main()
