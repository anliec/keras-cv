import glob
import os
import cv2
import multiprocessing
import numpy as np
import itertools
from tensorflow.python.keras.utils import Sequence
import random
from bisect import bisect_left
from math import ceil


class YoloDataLoader(Sequence):
    def __init__(self, images_file_list, batch_size: int, image_shape, annotation_shape, shuffle=True,
                 class_to_load=("0",)):
        self.image_list = images_file_list
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.annotation_shape = annotation_shape
        if shuffle:
            random.shuffle(self.image_list)
        self.loaded_array = [None] * (int(ceil(len(self.image_list) / self.batch_size)))
        self.class_to_load = list(class_to_load)
        self.class_count = len(self.class_to_load)

    def __len__(self):
        return int(np.ceil(len(self.image_list) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        if self.loaded_array[idx] is None:
            images = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            self.loaded_array[idx] = self.load_data_list(images)
        return self.loaded_array[idx]

    def load_data_list(self, image_path_list):
        data = [self.load_yolo_pair(i) for i in image_path_list]
        tuple_data = np.array([d[0] for d in data], dtype=np.float16), \
                     [np.array([d[1][0].reshape(d[1][0].shape + (1,)) for d in data], dtype=np.float16),
                      np.array([d[1][1].reshape(d[1][1].shape + (1,)) for d in data], dtype=np.float16)]
        return tuple_data

    def load_yolo_gt(self, file_path: str, ):
        bb_coordinates = []
        with open(file_path, 'r') as gt:
            for line in gt:
                vals = line.split(" ")
                if vals[0] in self.class_to_load:
                    x, y, w, h = [float(v) for v in vals[1:]]
                    c = self.class_to_load.index(vals[0])
                    bb_coordinates.append((x, y, w, h, c))
        return self.get_annotation_from_yolo_gt_values(bb_coordinates)

    def get_annotation_from_yolo_gt_values(self, bounding_box_coordinates_list):
        scores = np.zeros(self.annotation_shape, dtype=np.float16)
        sizes = np.zeros(self.annotation_shape, dtype=np.float16)
        for x, y, w, h, c in bounding_box_coordinates_list:
            x, w = [int(round(v * self.annotation_shape[1])) for v in [x, w]]
            y, h = [int(round(v * self.annotation_shape[0])) for v in [y, h]]
            scores[y, x] = 1.0
            sizes[y, x] = (h + w) / 2.0
        return scores, sizes

    def load_yolo_image(self, image_path: str):
        im = cv2.imread(image_path)
        im = cv2.resize(im, self.image_shape[::-1])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float16)
        im /= 255
        return im

    def load_yolo_pair(self, path_to_image: str):
        path_to_annotation = os.path.splitext(path_to_image)[0] + ".txt"
        gt = self.load_yolo_gt(path_to_annotation)
        im = self.load_yolo_image(path_to_image)
        return im, gt

    # def load_yolo_pair_for_map(self, args):
    #     return self.load_yolo_pair(args[0], args[1], args[2])
    #
    # def load_annotations_from_dir(self, dir_path: str, image_shape, annotation_shape, images_regex: str = "*.jpg"):
    #     images = glob.glob(os.path.join(dir_path, images_regex))
    #     print("{} images found in {}".format(len(images), dir_path))
    #     x = np.zeros(shape=(len(images),) + tuple(image_shape) + (3,), dtype=np.float)
    #     y_score = np.zeros(shape=(len(images),) + tuple(annotation_shape) + (1,), dtype=np.float)
    #     y_size = np.zeros(shape=(len(images),) + tuple(annotation_shape) + (1,), dtype=np.float)
    #     pool = multiprocessing.Pool()
    #     dataloader = pool.imap_unordered(self.load_yolo_pair_for_map,
    #                                      zip(images, itertools.repeat(image_shape), itertools.repeat(annotation_shape)),
    #                                      chunksize=25)
    #     for i, (im, score, size) in enumerate(dataloader):
    #         x[i, :, :, :] = im
    #         y_score[i, :, :, 0] = score
    #         y_size[i, :, :, 0] = size
    #
    #     return x, (y_score, y_size)

    def data_list_iterator(self):
        for image in self.image_list:
            yield self.load_yolo_pair(image)


def take_closest_index(my_list: list, my_number: float) -> int:
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(my_list, my_number)
    if pos == 0:
        return pos
    elif pos == len(my_list):
        return pos - 1
    before = my_list[pos - 1]
    after = my_list[pos]
    if after - my_number < my_number - before:
        return pos
    else:
        return pos - 1


class RawYoloDataLoader(YoloDataLoader):
    """
    Load yolo data generating a 3D matrix ground truth (position and size probability)
    """
    def __init__(self, images_file_list, batch_size: int, image_shape, annotation_shape, pyramid_size_list: list,
                 shuffle=True):
        super().__init__(images_file_list, batch_size, image_shape, annotation_shape, shuffle)
        self.pyramid_size_list = pyramid_size_list

    def load_data_list(self, image_path_list):
        data = [self.load_yolo_pair(i) for i in image_path_list]
        return np.array([d[0] for d in data], dtype=np.float16), np.array([d[1] for d in data], dtype=np.float16)

    def get_annotation_from_yolo_gt_values(self, bounding_box_coordinates_list):
        raw = np.zeros(self.annotation_shape + (len(self.pyramid_size_list), 1), dtype=np.float16)
        for x, y, w, h, c in bounding_box_coordinates_list:
            x, w = [int(round(v * self.annotation_shape[1])) for v in [x, w]]
            y, h = [int(round(v * self.annotation_shape[0])) for v in [y, h]]
            index = int(take_closest_index(self.pyramid_size_list, (h + w) / 2.0))
            raw[y, x, index, 0] = 1.0
        return raw


def list_data_from_dir(dir_path: str, images_regex: str = "*.jpg"):
    images = glob.glob(os.path.join(dir_path, images_regex))
    images = [i for i in images if os.path.isfile(os.path.splitext(i)[0] + ".txt")]
    return images


class SSDLikeYoloDataLoader(YoloDataLoader):
    """
    Load yolo data generating a 2D matrix ground truth (linearised position x class)
    """
    def __init__(self, images_file_list, batch_size: int, image_shape, annotation_shapes: list, pyramid_size_list: list,
                 shuffle=True, class_to_load=("0",)):
        super().__init__(images_file_list, batch_size, image_shape, annotation_shapes, shuffle, class_to_load)
        self.pyramid_size_list = pyramid_size_list
        if len(pyramid_size_list) % len(annotation_shapes) != 0:
            raise ValueError("The provided size and shapes are not compatible, please provide a correctly sized list"
                             "(ex: list of the same size)")
        self.num_boxes = sum([x * y for x, y in annotation_shapes])
        self.class_count += 1  # add background class
        self.size_per_prediction_shape = int(len(pyramid_size_list) // len(annotation_shapes))

    def load_data_list(self, image_path_list):
        data = [self.load_yolo_pair(i) for i in image_path_list]
        return np.array([d[0] for d in data], dtype=np.float16), np.array([d[1] for d in data], dtype=np.float16)

    def get_annotation_from_yolo_gt_values(self, bounding_box_coordinates_list):
        raws = [np.zeros(shape=list(s.astype(np.int)) + [self.class_count], dtype=np.float16)
                for s in self.annotation_shape]
        # set the background class to one on every predictions
        for r in raws:
            r[:, :, 0] = 1.0
        # change the value at the bounding boxes coordinates
        for x, y, w, h, c in bounding_box_coordinates_list:
            w = int(round(w * self.image_shape[1]))
            h = int(round(h * self.image_shape[0]))
            size_index = int(take_closest_index(self.pyramid_size_list, (h + w) / 2.0))
            shape_index = int(size_index // self.size_per_prediction_shape)
            shape = self.annotation_shape[shape_index]
            x = int(round(x * shape[1]))
            y = int(round(y * shape[0]))
            raws[shape_index][y, x, c + 1] = 1.0
            raws[shape_index][y, x, 0] = 0.0
        concat_flatten_raws = np.concatenate([a.reshape(-1) for a in raws])
        return concat_flatten_raws








