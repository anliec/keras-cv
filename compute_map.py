import tensorflow as tf

from load_yolo_data import YoloDataLoader
from detection_processing import DetectionProcessor


def eval_map(validation_data: YoloDataLoader, model: tf.keras.Model, processor: DetectionProcessor,
             iou_thresholds=(0.5,)):
    fn_roi = []
    matches = []
    min_iou_th = min(iou_thresholds)

    for i in range(len(validation_data)):
        x_val, y_true = validation_data[i]
        y_pred = model.predict(x_val)
        y_pred_roi = processor.process_detection(y_pred)
        y_true_roi = processor.process_detection(y_true)
        for im_y_pred_roi, im_y_true_roi in zip(y_pred_roi, y_true_roi):
            for pred_roi in sorted(im_y_pred_roi, key=lambda roi: -roi.confidence):
                best = None
                best_iou = min_iou_th
                best_i = None
                for true_roi_index, true_roi in enumerate(im_y_true_roi):
                    iou = pred_roi.get_overlap(true_roi)
                    if iou > best_iou:
                        best = true_roi
                        best_iou = iou
                        best_i = true_roi_index
                if best is not None:
                    matches.append((pred_roi, best, best_iou))
                    del im_y_true_roi[best_i]
                else:
                    matches.append((pred_roi, None, None))
            fn_roi += im_y_true_roi

    gt_count = len([tp for tp in matches if tp[1] is not None]) + len(fn_roi)
    rolling_tp_counts = {k: [] for k in iou_thresholds}
    rolling_fp_counts = {k: [] for k in iou_thresholds}
    rolling_precision = {k: [] for k in iou_thresholds}
    rolling_recall = {k: [] for k in iou_thresholds}
    last_tp, last_fp = {k: 0 for k in iou_thresholds}, {k: 0 for k in iou_thresholds}
    for pred_roi, true_roi, iou in sorted(matches, key=lambda m: -m[0].confidence):
        for iou_th in iou_thresholds:
            if iou >= iou_th and true_roi is not None:
                last_tp[iou_th] += 1
            else:
                last_fp[iou_th] += 1

            rolling_tp_counts[iou_th].append(last_tp[iou_th])
            rolling_fp_counts[iou_th].append(last_fp[iou_th])

            precision = last_tp[iou_th] / (last_fp[iou_th] + last_tp[iou_th])
            rolling_precision[iou_th].append(precision)
            rolling_recall[iou_th].append(last_tp[iou_th] / gt_count)

    auc = {}
    for iou_th in iou_thresholds:
        top_precision_by_recall = [(0, 1.0)]
        last_top_precision = 0
        for p, r in zip(rolling_precision[iou_th][::-1], rolling_recall[iou_th][::-1]):
            if last_top_precision < p:
                top_precision_by_recall.append((p, r))
                last_top_precision = p

        top_precision_by_recall.append((last_top_precision, 0.0))

        area_under_curve = sum([p_next[1] * (p_next[0] - p_prev[0]) for p_prev, p_next in
                                zip(top_precision_by_recall[:-1], top_precision_by_recall[1:])])
        auc[iou_th] = area_under_curve

    fn_counts = {k: gt_count - v for k, v in last_tp.items()}
    return auc, last_tp, last_fp, fn_counts
