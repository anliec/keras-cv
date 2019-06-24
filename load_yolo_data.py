import glob
import os
import cv2
import multiprocessing
import numpy as np
import itertools


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


