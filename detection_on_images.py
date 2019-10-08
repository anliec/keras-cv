import cv2
import numpy as np
from glob import glob
import os
import datetime
import tensorflow as tf
import json
import math

from detection_processing import process_detection_raw
from load_network import load_network


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, image_size, images_path: str, batch_size: int, image_extension: str = ".jpg"):
        super().__init__()
        self.images_list = glob(os.path.join(images_path, "*" + image_extension))
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images_list) / self.batch_size)

    def load_yolo_image(self, image_path: str):
        im = cv2.imread(image_path)
        im = cv2.resize(im, self.image_size[::-1])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float16)
        im /= 255
        return im

    def __getitem__(self, idx):
        i = self.batch_size * idx
        return np.array([self.load_yolo_image(p) for p in self.images_list[i:i + self.batch_size]], dtype=np.float16)


def run_model_on_images(model: tf.keras.Model, sizes: list, images_path: str, output_file_path: str = None,
                        image_extension: str = ".jpg", batch_size: int = 32, threshold: float = 0.5):
    test_sequence = ImageSequence(model.input_shape[1:3], images_path, batch_size, image_extension)

    detection = []
    step_size = 100
    for i in range(0, len(test_sequence), step_size):
        print("Batch {} / {}".format(int(i / step_size), math.ceil(len(test_sequence) / step_size)))
        prediction = model.predict_generator(test_sequence, steps=min(step_size, len(test_sequence) - i), verbose=1)

        pred_roi = process_detection_raw(prediction, sizes, threshold)

        detection += [{"frame_number": os.path.basename(image_path),
                       "signs": [{"coordinates": [roi.X, roi.Y, roi.W, roi.H], "class": "Unique"} for roi in rois]}
                      for rois, image_path in zip(pred_roi, test_sequence.images_list)]

    detection = sorted(detection, key=lambda e: e["frame_number"])

    output = {
        "output": {
            "frame_cfg": {
                "dir": os.path.realpath(images_path),
            },
            "framework": {
                "name": "Nnet",
                "version": "Alpha 0",
                "test_date": datetime.datetime.now().strftime("%A %d. %B %Y"),
            },
            "frames": detection
        }
    }

    if output_file_path is not None:
        with open(output_file_path, 'w') as f:
            json.dump(output, f)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='Path to the input training data')
    parser.add_argument('model_path',
                        type=str,
                        help='Path to the md5 model')
    parser.add_argument('model_sizes',
                        type=float,
                        nargs='+',
                        help='size of the different layers of the model')
    parser.add_argument('output_file_path',
                        type=str,
                        help='Path to the input training data')
    parser.add_argument('-ext', '--image-extension',
                        dest='image_extension',
                        type=str,
                        default="*.jpg",
                        help='extension of the image to look for')
    parser.add_argument('-b', '--batch-size',
                        required=False,
                        type=int,
                        default=32,
                        help='Size a of batch of data send to the neural network at one time',
                        dest="batch")
    parser.add_argument('-t', '--threshold',
                        required=False,
                        type=float,
                        default=0.5,
                        help='threshold for considering an output as a detection',
                        dest="threshold")
    args = parser.parse_args()

    model, sizes = load_network(size_value=[226, 402], random_init=True, pyramid_depth=4,
                                first_pyramid_output=0, add_noise=False)

    model.load_weights(args.model_path)

    run_model_on_images(model, sizes, args.data_path, args.output_file_path,
                        image_extension=args.image_extension, threshold=args.threshold, batch_size=args.batch)










