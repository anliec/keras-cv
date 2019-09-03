from tensorflow.python.keras import backend as K
import numpy as np
from itertools import product


from detection_processing import process_detection_raw
from utils import gaussian_kernel, gaussian_kernel_3d


def map_metric(concentration_kernel_size: int = 7, concentration_kernel_height: int = 3):

    k = -gaussian_kernel_3d(concentration_kernel_size, concentration_kernel_height)
    # k = -np.ones(shape=(concentration_kernel_size, concentration_kernel_size, concentration_kernel_height))
    center_x_y = int(concentration_kernel_size // 2 + 1)
    center_h = int(concentration_kernel_height // 2 + 1)
    k[center_x_y, center_x_y, center_h] = abs(np.sum(k) - k[center_x_y, center_x_y, center_h]) / 2
    k = k.reshape((concentration_kernel_size, concentration_kernel_size, concentration_kernel_height, 1, 1))
    k = K.constant(k, dtype=np.float32)

    def precision(y_true, y_pred):
        # filter close detections
        y_pred_filtered = K.conv3d(y_pred, k, padding="same")

        # tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted = K.sum(K.round(K.clip(y_pred_filtered, 0, 1)))

        # p = tp / (predicted + K.epsilon())
        return predicted

    def gt_count(y_true, y_pred):
        return K.sum(K.round(K.clip(y_true, 0, 1)))

    return precision






