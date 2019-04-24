import numpy as np

from detection_processing import process_detection


def recall(overlap: float, detection_threshold: float):

    def metric(y_true, y_pred):
        true_score, true_size = y_true
        pred_score, pred_size = y_pred

        detection_boxes = process_detection(pred_score, pred_size, detection_threshold)
        true_boxes = process_detection(true_score, true_size, detection_threshold)

        fp = 0

        for detection_roi in detection_boxes:
            for true_roi in true_boxes:
                if detection_roi.get_overlap(true_roi) > overlap:
                    break
            else:
                fp += 1

        tp = len(detection_boxes) - fp
        # fn = len(true_boxes) - tp

        return tp / len(true_boxes)

    return metric





