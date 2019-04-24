import numpy as np


class Roi(object):
    def __init__(self, confidence: float, center_pos, size):
        w = size
        h = size
        self.X = int(round(center_pos[0] - w / 2))
        self.Y = int(round(center_pos[1] - h / 2))
        if self.X < 0:
            w += self.X
            self.X = 0
        if self.Y < 0:
            h += self.Y
            self.Y = 0
        self.W = int(round(w))
        self.H = int(round(h))
        self.confidence = confidence

    def get_overlap(self, other):
        return get_overlap(self, other)


def get_overlap(roi_gt, roi_detection):
    # first check that both roi are intersecting
    if not (roi_gt.X < roi_detection.X + roi_detection.W and roi_gt.X + roi_gt.W > roi_detection.X and
            roi_gt.Y < roi_detection.Y + roi_detection.H and roi_gt.Y + roi_gt.H > roi_detection.Y):
        return 0.0
    # compute the area of intersection rectangle
    inter_width = min(filter(lambda v: v > 0, [roi_gt.X + roi_gt.W - roi_detection.X,
                                               roi_detection.X + roi_detection.W - roi_gt.X,
                                               roi_detection.W, roi_gt.W]))
    inter_height = min(filter(lambda v: v > 0, [roi_gt.Y + roi_gt.H - roi_detection.Y,
                                                roi_detection.Y + roi_detection.H - roi_gt.Y,
                                                roi_detection.H, roi_gt.H]))
    inter_area = inter_width * inter_height
    # compute the area of both the detection and ground-truth
    # rectangles
    gt_area = roi_gt.W * roi_gt.H
    det_area = roi_detection.W * roi_detection.H
    union_area = gt_area + det_area - inter_area
    # compute the intersection over union
    try:
        overlap = inter_area * 100.0 / union_area
    except ZeroDivisionError:
        overlap = 0.0
    # return the intersection over union value
    return overlap


def process_detection(score: np.ndarray, size: np.ndarray, threshold: float = 0.5):
    results = []
    while True:
        pos = np.argmax(score)
        confidence = score[pos]
        if confidence < threshold:
            break
        else:
            b = Roi(confidence, pos, size[pos])
            results.append(b)
            # mask out selected box
            score[b.X:b.X + b.W, b.Y:b.Y + b.H] = 0.0
    return results





