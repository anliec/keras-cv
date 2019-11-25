import cv2
import numpy as np
import os
import datetime
import tensorflow as tf
import json
import math

from detection_processing import DetectionProcessor
from load_network import load_network

RGB_AVERAGE = np.array([100.27196761, 117.94775357, 130.05339633], dtype=np.float32)
RGB_STD = np.array([36.48646844, 27.12285032, 27.58063623], dtype=np.float32)


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(self, path_to_video: str, batch_size: int, image_shape):
        super().__init__()
        self.cap = cv2.VideoCapture(path_to_video)
        self.batch_size = batch_size
        self.length = int(math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / batch_size))
        self.has_next = self.length > 0
        self.image_shape = image_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_image_list = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.batch_size)
        for i in range(self.batch_size):
            success, image = self.cap.read()
            if not success:
                break
            image = self.preprocess_image(image, image_shape=self.image_shape, normalize=True)
            batch_image_list.append(image)
        return np.array(batch_image_list, dtype=np.float16)

    @staticmethod
    def preprocess_image(im: np.ndarray, image_shape, normalize: bool = True):
        im = cv2.resize(im, image_shape[::-1])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32)
        if normalize:
            im -= RGB_AVERAGE
            im /= RGB_STD
        return im


def run_model_on_images(model: tf.keras.Model, sizes: list, shapes: list, images_path: str,
                        output_file_path: str = None, batch_size: int = 32,
                        threshold: float = 0.5, nms_threshold: float = 0.5):
    test_sequence = VideoSequence(path_to_video=images_path, image_shape=model.input_shape[1:3], batch_size=batch_size)

    detection_processor = DetectionProcessor(sizes=sizes, shapes=shapes, image_size=model.input_shape[1:3],
                                             threshold=threshold, nms_threshold=nms_threshold)

    detection = []
    step_size = 100
    for i in range(0, len(test_sequence), step_size):
        print("Batch {} / {}:".format(int(i / step_size), math.ceil(len(test_sequence) / step_size)))
        prediction = model.predict_generator(test_sequence, steps=min(step_size, len(test_sequence) - i), verbose=1)

        pred_roi = detection_processor.process_detection(prediction, pool=None)

        detection += [{"frame_number": os.path.basename(image_path),
                       "signs": [{"coordinates": [int(roi.X), int(roi.Y), int(roi.W), int(roi.H)],
                                  "class": "{}".format(roi.c),
                                  "detection_confidence": float(roi.confidence)} for roi in rois]}
                      for rois, image_path in zip(pred_roi, test_sequence.images_list)]

    detection = sorted(detection, key=lambda e: e["frame_number"])

    output = {
        "output": {
            "frame_cfg": {
                "dir": os.path.realpath(images_path),
            },
            "framework": {
                "name": "Nnet",
                "version": "Alpha 1",
                "test_date": datetime.datetime.now().strftime("%A %d. %B %Y"),
            },
            "frames": detection
        }
    }

    if output_file_path is not None:
        with open(output_file_path, 'w') as f:
            json.dump(output, f, indent=4)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='Path to the input video')
    parser.add_argument('model_path',
                        type=str,
                        help='Path to the md5 model')
    parser.add_argument('config_path',
                        type=str,
                        help='Path to the json model config file')
    parser.add_argument('output_file_path',
                        type=str,
                        help='Path to the output json file')
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

    with open(args.config_path, 'r') as j:
        config = json.load(j)

    model, sizes, shapes = load_network(**config)

    model.load_weights(args.model_path)

    run_model_on_images(model, sizes, shapes, args.data_path, args.output_file_path, threshold=args.threshold,
                        batch_size=args.batch)











