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
        # if self.X < 0:
        #     self.W += self.X
        #     self.X = 0
        # if self.Y < 0:
        #     self.H += self.Y
        #     self.Y = 0
        self.confidence = confidence

    def get_overlap(self, other):
        return get_overlap(self, other)

    def up_left_corner(self):
        return int(round(self.Y)), int(round(self.X))

    def down_right_corner(self):
        return int(round(self.Y + self.H)), int(round(self.X + self.W))

    def print(self):
        print("X: {} Y: {} W: {}".format(self.X, self.Y, self.W))


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
        overlap = inter_area / union_area
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


class DetectionProcessor:
    """
    Class that  handle the conversion from a output head of nNet into a bounding box.
    """
    def __init__(self, sizes: list, shapes: list, image_size, threshold: float = 0.5, nms_threshold: float = 0.5):
        self.shapes = shapes
        self.sizes = sizes
        self.threshold = threshold
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.linear_index_shapes = [0]
        i = 0
        for w, h in self.shapes:
            i += int(w * h)
            self.linear_index_shapes.append(i)
        if len(shapes) != len(sizes):
            assert len(sizes) % len(shapes) == 0
            i = len(sizes) // len(shapes)
            self.sizes = self.sizes[::i]

    def _unravel_index(self, index):
        for i, v in enumerate(self.linear_index_shapes):
            if v > index:
                shape_index = i - 1
                break
        else:
            raise ValueError("unexpected index value given")
        linear_index_in_shape = index - self.linear_index_shapes[shape_index]
        size = self.sizes[shape_index]
        pos = np.array(np.unravel_index(linear_index_in_shape, self.shapes[shape_index]), dtype=np.float32)
        # move the pos to the center of the pixel and scale it, then move to the top left corner
        pos[0] = ((pos[0] + 0.5) * (self.image_size[0] / self.shapes[shape_index][0]))
        pos[1] = ((pos[1] + 0.5) * (self.image_size[1] / self.shapes[shape_index][1]))
        return pos, size

    def _pos_to_roi(self, pos, conf: float):
        coord, size = self._unravel_index(pos[0])
        return Roi(conf, coord, size, pos[1], shape=self.image_size)

    def _process_image_detection(self, raw: np.ndarray):
        pos = np.where(raw[:, 1:] > self.threshold)
        prediction = [self._pos_to_roi(p, raw[p]) for p in zip(*pos)]
        if self.nms_threshold > 1.0:
            return prediction
        else:
            return self._non_max_suppression(prediction)

    def _non_max_suppression(self, prediction: list):
        i1, i2 = 0, 0
        # sort over confidence first !
        # faster algorithm possible !
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
        """
        Process the raw output given by nNet into actual bounding boxes
        @param raw: Output of the nNet network
        @param pool: multiprocessing pool to use to parallelize the process, set to None to run on current thread
        @return: A list of list of Roi object. First list as the same number of element than raw (i.e. number of image
        with detection) the second list is the number of detection for this output, formatted in a Roi object.
        """
        if pool is None:
            return [self._process_image_detection(pred) for pred in raw]
        else:
            return [roi for roi in pool.map(self._process_image_detection, raw)]







