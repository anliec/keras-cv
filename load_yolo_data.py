import glob
import os
import cv2
import multiprocessing
import numpy as np
import itertools
from keras.utils import Sequence
import random


class YoloDataLoader(Sequence):
    def __init__(self, images_file_list, batch_size: int, image_shape, annotation_shape, shuffle=True):
        self.image_list = images_file_list
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.annotation_shape = annotation_shape
        if shuffle:
            random.shuffle(self.image_list)

    def __len__(self):
        return int(np.ceil(len(self.image_list) / float(self.batch_size)))

    def __getitem__(self, idx: int):
        images = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = [load_yolo_pair(i, self.image_shape, self.annotation_shape) for i in images]
        return np.array([d[0] for d in data]), \
               [np.array([d[1].reshape(d[1].shape + (1,)) for d in data]),
                np.array([d[2].reshape(d[2].shape + (1,)) for d in data])]


def load_yolo_gt(file_path: str, out_shape, class_to_load=("0",)):
    scores = np.zeros(out_shape)
    sizes = np.zeros(out_shape)
    with open(file_path, 'r') as gt:
        for line in gt:
            vals = line.split(" ")
            if vals[0] in class_to_load:
                x, y, w, h = [float(v) for v in vals[1:]]
                x, w = [int(round(v * out_shape[1])) for v in [x, w]]
                y, h = [int(round(v * out_shape[0])) for v in [y, h]]
                scores[y, x] = 1.0
                sizes[y, x] = (h + w) / 2.0
    return scores, sizes


def load_yolo_image(image_path: str, input_size):
    im = cv2.imread(image_path)
    im = cv2.resize(im, input_size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float)
    im /= im.max()
    return im


def load_yolo_pair(path_to_image: str, image_shape, annotation_shape):
    path_to_annotation = os.path.splitext(path_to_image)[0] + ".txt"
    scores, sizes = load_yolo_gt(path_to_annotation, annotation_shape)
    im = load_yolo_image(path_to_image, tuple(image_shape[::-1]))
    return im, scores, sizes


def load_yolo_pair_for_map(args):
    return load_yolo_pair(args[0], args[1], args[2])


def load_annotations_from_dir(dir_path: str, image_shape, annotation_shape, images_regex: str = "*.jpg"):
    images = glob.glob(os.path.join(dir_path, images_regex))
    print("{} images found in {}".format(len(images), dir_path))
    x = np.zeros(shape=(len(images),) + tuple(image_shape) + (3,), dtype=np.float)
    y_score = np.zeros(shape=(len(images),) + tuple(annotation_shape) + (1,), dtype=np.float)
    y_size = np.zeros(shape=(len(images),) + tuple(annotation_shape) + (1,), dtype=np.float)

    pool = multiprocessing.Pool()

    dataloader = pool.imap_unordered(load_yolo_pair_for_map,
                                     zip(images, itertools.repeat(image_shape), itertools.repeat(annotation_shape)),
                                     chunksize=25)

    for i, (im, score, size) in enumerate(dataloader):
        x[i, :, :, :] = im
        y_score[i, :, :, 0] = score
        y_size[i, :, :, 0] = size

    return x, (y_score, y_size)


def list_data_from_dir(dir_path: str, images_regex: str = "*.jpg"):
    images = glob.glob(os.path.join(dir_path, images_regex))
    images = [i for i in images if os.path.isfile(os.path.splitext(i)[0] + ".txt")]
    return images


def data_list_iterator(images_file_list, image_shape, annotation_shape):
    for image in images_file_list:
        yield load_yolo_pair(image, image_shape, annotation_shape)




