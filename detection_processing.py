import numpy as np
import cv2
import multiprocessing


class Roi(object):
    def __init__(self, confidence: float, center_pos, size, class_name=0, shape=None):
        self.W = size
        self.H = size
        self.X = center_pos[0] - self.W / 2
        self.Y = center_pos[1] - self.H / 2
        self.c = class_name
        self.shape = shape
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
    assert raw.shape[3] == len(sizes)

    per_image_results = []
    for det in raw:
        results = []

        while True:
            pos = np.unravel_index(np.argmax(det), det.shape)
            confidence = det[pos]
            if confidence < threshold:
                break
            else:
                b = Roi(confidence, pos[0:2], sizes[pos[2]])
                b.shape = det.shape[0:2]
                results.append(b)
                # mask out selected box
                det[int(b.X):int(b.X + b.W), int(b.Y):int(b.Y + b.H), :, 0] = 0.0
        per_image_results.append(results)

    if len(per_image_results) == 1:
        return per_image_results[0]
    else:
        return per_image_results


class DetectionProcessor:
    def __init__(self, sizes: list, shapes: list, image_size, threshold: float = 0.5, nms_threshold: float = 0.5):
        self.shapes = shapes
        self.sizes = sizes
        self.threshold = threshold
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.linear_index_shapes = []
        i = 0
        for w, h in self.shapes:
            i += w * h
            self.linear_index_shapes.append(i)
        if len(shapes) != len(sizes):
            assert len(sizes) % len(shapes) == 0
            i = len(sizes) // len(shapes)
            self.sizes = self.sizes[::i]

    def unravel_index(self, index):
        for i, v in enumerate(self.linear_index_shapes):
            if v > index:
                shape_index = v
                break
        else:
            raise ValueError("unexpected index value given")
        linear_index_in_shape = index - self.linear_index_shapes[shape_index - 1]
        pos = np.unravel_index(linear_index_in_shape, self.shapes[shape_index])
        pos = pos * self.image_size[0] / self.shapes[shape_index][0]
        return pos, self.sizes[shape_index]

    def pos_to_roi(self, pos, conf: float):
        coord, size = self.unravel_index(pos[0])
        return Roi(conf, coord, size, pos[1], shape=self.image_size)

    def process_image_detection(self, raw: np.ndarray):
        pos = np.where(raw[:, 1:] > self.threshold)
        prediction = [self.pos_to_roi(p, raw[p]) for p in zip(*pos)]
        return self.non_max_suppression(prediction)

    def non_max_suppression(self, prediction: list):
        i1, i2 = 0, 0
        while i1 < len(prediction):
            p1 = prediction[i1]
            i2 = i1 + 1
            while i2 < len(prediction):
                p2 = prediction[i2]
                if p1.get_overlap(p2) > self.nms_threshold:
                    if p1.confidence > p2.confidence:
                        prediction.pop(i2)
                        i2 -= 1
                    else:
                        prediction.pop(i1)
                        i1 -= 1
                        break
                i2 += 1
            i1 += 1
        return prediction

    def process_detection(self, raw: np.ndarray, pool: multiprocessing.Pool = None):
        if pool is None:
            return map(self.process_detection, raw)
        else:
            return list(pool.map(self.process_detection, raw))







