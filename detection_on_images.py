import cv2
import numpy as np
from glob import glob
import os
import datetime
import tensorflow as tf

from detection_processing import process_detection_raw


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, image_size, images_path: str, batch_size: int, image_extension: str = ".jpg"):
        super().__init__()
        self.images_list = glob(os.path.join(images_path, "*" + image_extension))
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images_list)

    def load_yolo_image(self, image_path: str):
        im = cv2.imread(image_path)
        im = cv2.resize(im, self.image_shape[::-1])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float16)
        im /= 255
        return im

    def __getitem__(self, idx):
        i = self.batch_size * idx
        return [self.load_yolo_image(p) for p in self.images_list[i:i + self.batch_size]]


def run_model_on_images(model: tf.keras.Model, sizes: list, images_path: str, image_extension: str = ".jpg",
                        batch_size: int = 32, threshold: float = 0.5):
    test_sequence = ImageSequence(model.input_shape[1:3], images_path, batch_size, image_extension)

    prediction = model.predict_generator(test_sequence, use_multiprocessing=True, workers=4)

    pred_roi = process_detection_raw(prediction, sizes, threshold)

    detection = [{"frame_number": os.path.basename(image_path),
                      "signs": [{"coodinates": [roi.X, roi.Y, roi.W, roi.H], "class": "Unique"} for roi in rois]}
                 for rois, image_path in zip(pred_roi, test_sequence.images_list)]

    output = {
        "output": {
            "frame_cfg": {
                "dir": os.path.realpath(image_path),
            },
            "framework": {
                "name": "Nnet",
                "version": "Alpha 0",
                "test_date": datetime.datetime().strftime("%A %d. %B %Y"),
                "weights": "yolo3-tiny_gtsdb_final.weights"
            },
            "frames": detection
        }

    for rois, image_path in zip(pred_roi, test_sequence.images_list):
        image_dict = {"frame_number": os.path.basename(image_path),
                      "signs": [{"coodinates": [roi.X, roi.Y, roi.W, roi.H], "class": "Unique"} for roi in rois]}

        for roi in rois:
            s =







