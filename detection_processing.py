import numpy as np
import cv2


class Roi(object):
    def __init__(self, confidence: float, center_pos, size):
        self.W = size
        self.H = size
        self.X = center_pos[0] - self.W / 2
        self.Y = center_pos[1] - self.H / 2
        if self.X < 0:
            self.W += self.X
            self.X = 0
        if self.Y < 0:
            self.H += self.Y
            self.Y = 0
        self.confidence = confidence

    def get_overlap(self, other):
        return get_overlap(self, other)

    def up_left_corner(self):
        return int(round(self.Y)), int(round(self.X))

    def down_right_corner(self):
        return int(round(self.Y + self.H)), int(round(self.X + self.W))

    def print(self):
        print("X; {} Y: {} W: {}".format(self.X, self.Y, self.W))


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


def rescale_roi(roi: Roi, new_shape):
    fx = new_shape[0] / roi.shape[0]
    fy = new_shape[1] / roi.shape[1]
    roi.X *= fx
    roi.W *= fx
    roi.Y *= fy
    roi.H *= fy
    roi.shape = new_shape
    return roi


def process_detection(score: np.ndarray, size: np.ndarray, threshold: float = 0.5):
    results = []
    while True:
        pos = np.unravel_index(np.argmax(score), score.shape)
        confidence = score[pos]
        # print("confidence: {}".format(confidence))
        if confidence < threshold:
            # print("break")
            break
        else:
            b = Roi(confidence, pos[1:3], size[pos])
            b.shape = score.shape[1:3]
            results.append(b)
            # mask out selected box
            score[0, int(b.X):int(b.X + b.W), int(b.Y):int(b.Y + b.H), 0] = 0.0
    return results


def draw_roi(img, roi_list, color=(0, 255, 0), width=4):
    for roi in roi_list:
        rescale_roi(roi, img.shape[0:2])
        img = cv2.rectangle(img, roi.up_left_corner(), roi.down_right_corner(), color, width)
    return img


def process_detection_raw(raw: np.ndarray, sizes: list, threshold: float = 0.5):
    results = []
    assert raw.shape[3] == len(sizes)
    while True:
        pos = np.unravel_index(np.argmax(raw), raw.shape)
        confidence = raw[pos]
        if confidence < threshold:
            break
        else:
            b = Roi(confidence, pos[1:3], sizes[pos[3]])
            b.shape = raw.shape[1:3]
            results.append(b)
            # mask out selected box
            raw[0, int(b.X):int(b.X + b.W), int(b.Y):int(b.Y + b.H), :, 0] = 0.0
    return results






