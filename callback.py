import tensorflow as tf
import numpy as np

from detection_processing import DetectionProcessor
from compute_map import eval_map


class MAP_eval(tf.keras.callbacks.Callback):
    def __init__(self, validation_data: tf.keras.utils.Sequence, sizes, shapes, image_size, frequency: int = 1,
                 detection_threshold: float = 0.5, mns_threshold: float = 0.5, iou_thresholds=(0.5,),
                 epoch_start: int = 0):
        self.validation_data = validation_data
        self.maps = []
        self.scores = []
        self.epochs = []
        self.iou_thresholds = iou_thresholds
        self.frequency = frequency
        self.epoch_start = epoch_start
        self.processor = DetectionProcessor(sizes, shapes, image_size, detection_threshold, mns_threshold)

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.epoch_start and epoch % self.frequency == self.frequency - 1:
            mean_ap, tp, fp, fn = eval_map(self.validation_data, self.model, self.processor, self.iou_thresholds)
            print()
            for th in self.iou_thresholds:
                print(" mAP@{}: {}  TP: {}  FP: {}  FN: {}".format(int(th * 100), mean_ap[th], tp[th], fp[th], fn[th]))
            self.maps.append(mean_ap)
            self.scores.append((tp, fp, fn))
            self.epochs.append(epoch)
