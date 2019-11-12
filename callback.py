import tensorflow as tf
import numpy as np

from detection_processing import Roi, DetectionProcessor


class MAP_eval(tf.keras.callbacks.Callback):
    def __init__(self, validation_data: tf.keras.utils.Sequence, sizes, shapes, image_size, frequency: int = 1,
                 detection_threshold: float = 0.5, mns_threshold: float = 0.5, iou_threshold: float = 0.5,
                 epoch_start: int = 0):
        self.validation_data = validation_data
        self.maps = []
        self.scores = []
        self.epochs = []
        self.iou_threshold = iou_threshold
        self.frequency = frequency
        self.epoch_start = epoch_start
        self.processor = DetectionProcessor(sizes, shapes, image_size, detection_threshold, mns_threshold)

    def eval_map(self):
        fn_roi = []
        matches = []
        for i in range(len(self.validation_data)):
            x_val, y_true = self.validation_data[i]
            y_pred = self.model.predict(x_val)
            y_pred_roi = self.processor.process_detection(y_pred)
            y_true_roi = self.processor.process_detection(y_true)
            for im_y_pred_roi, im_y_true_roi in zip(y_pred_roi, y_true_roi):
                for pred_roi in sorted(im_y_pred_roi, key=lambda roi: -roi.confidence):
                    best = None
                    best_iou = self.iou_threshold
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
        rolling_tp_counts = []
        rolling_fp_counts = []
        rolling_precision = []
        rolling_recall = []
        last_tp, last_fp = 0, 0
        for pred_roi, true_roi, iou in sorted(matches, key=lambda m: -m[0].confidence):
            if true_roi is not None:
                last_tp += 1
            else:
                last_fp += 1

            rolling_tp_counts.append(last_tp)
            rolling_fp_counts.append(last_fp)

            rolling_precision.append(last_tp / (last_fp + last_tp))
            rolling_recall.append(last_tp / gt_count)

        top_precision_by_recall = [(0, 1.0)]
        last_top_precision = 0
        for p, r in zip(rolling_precision[::-1], rolling_recall[::-1]):
            if last_top_precision < p:
                top_precision_by_recall.append((p, r))
                last_top_precision = p

        top_precision_by_recall.append((last_top_precision, 0.0))

        area_under_curve = sum([p_next[1] * (p_next[0] - p_prev[0]) for p_prev, p_next in
                                zip(top_precision_by_recall[:-1], top_precision_by_recall[1:])])

        return area_under_curve, last_tp, last_fp, len(fn_roi)

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.epoch_start and epoch % self.frequency == 0:
            mean_ap, tp, fp, fn = self.eval_map()
            print(" mAP: {}  TP: {}  FP: {}  FN: {}".format(mean_ap, tp, fp, fn))
            self.maps.append(mean_ap)
            self.scores.append((tp, fp, fn))
            self.epochs.append(epoch)
