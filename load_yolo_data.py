import glob
import os
import cv2
import multiprocessing
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from bisect import bisect_left
from math import ceil
import tensorflow as tf
import itertools


RGB_AVERAGE = np.array([100.27196761, 117.94775357, 130.05339633], dtype=np.float32)
RGB_STD = np.array([36.48646844, 27.12285032, 27.58063623], dtype=np.float32)


"""
A set og function that load Yolo like annotations and make them available for training.
Main entry point: YoloDataLoader
"""


def read_yolo_image(image_path: str, image_shape, normalize: bool = True):
    im = cv2.imread(image_path)
    im = cv2.resize(im, image_shape[::-1])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    if normalize:
        im = normalize_image(im)
    return im


def normalize_image(im: np.ndarray):
    im -= RGB_AVERAGE
    im /= RGB_STD
    return im


class YoloBoundingBox:
    def __init__(self, definition_line: str):
        vals = definition_line.split(" ")
        self.x, self.y, self.w, self.h = [float(v) for v in vals[1:]]
        self.c = int(vals[0])
        self.annotation_x, self.annotation_y, self.annotation_w, self.annotation_h = self.x, self.y, self.w, self.h

    def reload_default_value(self):
        self.x, self.y, self.w, self.h = self.annotation_x, self.annotation_y, self.annotation_w, self.annotation_h

    def get_pixel_coordinates(self, image_size):
        x, w = [int(round(v * image_size[1])) for v in [self.x, self.w]]
        y, h = [int(round(v * image_size[0])) for v in [self.y, self.h]]
        return x, y, w, h

    def transform(self, transform, image_shape):
        """
        Apply the given transform to the bounding box. Transform regarding to the original value, not the last
        transformation.
        The transformation is done in the following order: zoom, move and then flip
        :param transform: a dict describing the transformation to apply (see https://keras.io/preprocessing/image/
        for more information on this dict content)
        :param image_shape: shape of the image corresponding to the transform (to get relative tx, ty)
        """
        self.x = 0.5 + ((self.annotation_x - (transform['ty'] / image_shape[1]) - 0.5) / transform['zy'])
        self.w = (self.annotation_w / transform['zy'])
        self.y = 0.5 + ((self.annotation_y - (transform['tx'] / image_shape[0]) - 0.5) / transform['zx'])
        self.h = (self.annotation_h / transform['zx'])
        if transform['flip_horizontal'] == 1:
            self.x = 1.0 - self.x


class YoloImageAnnotation:
    def __init__(self, annotation_file: str):
        with open(annotation_file, 'r') as gt:
            self.boxes = [YoloBoundingBox(line) for line in gt]

    def apply_transform(self, transform, image_shape):
        [b.transform(transform, image_shape) for b in self.boxes]


def augment_data(data, image_generator, image_shape):
    im, gt = data
    transform = image_generator.get_random_transform(image_shape)
    gt.apply_transform(transform, image_shape)
    im = normalize_image(image_generator.apply_transform(im, transform))
    return np.ascontiguousarray(im, dtype=np.float32), gt


class YoloDataLoader(Sequence):
    def __init__(self, images_file_list, batch_size: int, image_shape, annotation_shapes, pyramid_size_list: list,
                 shuffle=True, class_to_load=(0,), zoom_range=(1.0, 1.0), movement_range_width=0.0,
                 movement_range_height=0.0, flip: bool = False, disable_augmentation: bool = True,
                 brightness_range=None, use_multiprocessing: bool = False, pool: multiprocessing.Pool = None):
        self.image_list = images_file_list
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.annotation_shape = annotation_shapes
        if shuffle:
            random.shuffle(self.image_list)
        self.loaded_array = [[]] * (int(ceil(len(self.image_list) / self.batch_size)))
        self.class_to_load = set(class_to_load)
        self.class_count = len(self.class_to_load) + 1  # add background class
        self.pyramid_size_list = pyramid_size_list
        self.squared_pyramid_size_list = [s ** 2 for s in self.pyramid_size_list]
        if len(pyramid_size_list) % len(annotation_shapes) != 0:
            raise ValueError("The provided size and shapes are not compatible, please provide a correctly sized list"
                             "(ex: list of the same size)")
        self.num_boxes = sum([x * y for x, y in annotation_shapes])
        self.size_per_prediction_shape = int(len(pyramid_size_list) // len(annotation_shapes))
        self.use_multiprocessing = use_multiprocessing
        self.pool = pool
        # Image augmentation
        self.disable_augmentation = disable_augmentation
        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0,
            width_shift_range=movement_range_width,
            height_shift_range=movement_range_height,
            brightness_range=brightness_range,
            shear_range=0.,
            zoom_range=zoom_range,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=flip,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format='channels_last',
            validation_split=0.0,
            dtype='float32'
        )

    def __len__(self):
        return int(np.ceil(len(self.image_list) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        if len(self.loaded_array[idx]) == 0:
            images = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            self.loaded_array[idx] = self.load_data_list(images)
        if not self.disable_augmentation:
            if self.use_multiprocessing:
                data = list(self.pool.starmap(augment_data, zip(self.loaded_array[idx],
                                                                itertools.repeat(self.image_generator),
                                                                itertools.repeat(self.image_shape)),
                                              chunksize=int(self.batch_size / os.cpu_count()) + 1))
            else:
                data = [self.augment_data(im, gt) for im, gt in self.loaded_array[idx]]
        else:
            data = self.loaded_array[idx]
        return (np.array([d[0] for d in data], dtype=np.float16),
                np.array([self.get_annotation_from_yolo_gt_values(d[1]) for d in data], dtype=np.float16))

    def preload_data(self):
        print("Loading all the data to RAM...")
        for i, data in enumerate(self.loaded_array):
            print("{:3d}%".format(int(100 * i / len(self.loaded_array))), end='\r')
            if len(data) == 0:
                images = self.image_list[i * self.batch_size:(i + 1) * self.batch_size]
                self.loaded_array[i] = self.load_data_list(images)
        print()
        print("Data loaded!")

    def augment_data(self, im, gt: YoloImageAnnotation):
        transform = self.image_generator.get_random_transform(self.image_shape)
        gt.apply_transform(transform, self.image_shape)
        im = normalize_image(self.image_generator.apply_transform(im, transform))
        return np.ascontiguousarray(im, dtype=np.float32), gt

    def load_data_list(self, image_path_list):
        return [self.load_yolo_pair(i) for i in image_path_list]

    def load_yolo_pair(self, path_to_image: str):
        path_to_annotation = os.path.splitext(path_to_image)[0] + ".txt"
        gt = YoloImageAnnotation(path_to_annotation)
        im = read_yolo_image(path_to_image, self.image_shape, normalize=self.disable_augmentation)
        return im, gt

    def data_list_iterator(self):
        for i in range(self.__len__()):
            batch = self.__getitem__(i)
            for im, annotation in zip(*batch):
                yield im, annotation

    def get_annotation_from_yolo_gt_values(self, gt: YoloImageAnnotation):
        raws = [np.zeros(shape=list(s.astype(np.int)) + [self.class_count], dtype=np.float16)
                for s in self.annotation_shape]
        # set the background class to one on every predictions
        for r in raws:
            r[:, :, 0] = 1.0
        # change the value at the bounding boxes coordinates
        for b in gt.boxes:
            if b.c not in self.class_to_load:
                continue
            if 0 > b.x + b.w / 2 or 1.0 < b.x - b.w / 2 or 0 > b.y - b.h / 2 or 1.0 < b.y - b.h / 2:
                continue
            w = int(round(b.w * self.image_shape[1]))
            h = int(round(b.h * self.image_shape[0]))
            # chose the closest surface size index
            wh = w * h
            # if the box is too big or too small to be detected, discard it (IoU > 0.25)
            if wh * 4 < self.pyramid_size_list[0]:  # or wh > 4 * self.pyramid_size_list[-1]:
                continue
            size_index = int(take_closest_index(self.squared_pyramid_size_list, wh))
            shape_index = int(size_index // self.size_per_prediction_shape)
            shape = self.annotation_shape[shape_index]
            x = min(max(int(round((b.x * shape[1]) - 0.5)), 0), shape[1] - 1)
            y = min(max(int(round((b.y * shape[0]) - 0.5)), 0), shape[0] - 1)
            raws[shape_index][y, x, b.c + 1] = 1.0
            raws[shape_index][y, x, 0] = 0.0
        concat_flatten_raws = np.concatenate([a.reshape(-1, self.class_count) for a in raws], axis=0)
        return concat_flatten_raws


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


def list_data_from_dir(dir_path: str, images_regex: str = "*.jpg"):
    images = glob.glob(os.path.join(dir_path, images_regex))
    images = [i for i in images if os.path.isfile(os.path.splitext(i)[0] + ".txt")]
    return images
