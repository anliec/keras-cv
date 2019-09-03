from tensorflow.python.keras import backend as K
import numpy as np

from utils import gaussian_kernel, gaussian_kernel_3d


def detection_loss(gaussian_diameter: int = 31, score_tp_weight: float = 0.9, score_fp_weight: float = 0.5,
                   score_weight: float = 0.5, size_weight: float = 0.5, score_min_bound=0.0):
    g = np.zeros(shape=(gaussian_diameter, gaussian_diameter, 1, 1), dtype=np.float32)
    g[:, :, 0, 0] = gaussian_kernel(gaussian_diameter)
    g /= g.max()
    g = K.constant(g, dtype=np.float32)

    def score_loss(y_true, y_pred):
        score_upper_bound_target = K.conv2d(y_true, g, padding="same")
        score_upper_bound_target = K.clip(score_upper_bound_target, score_min_bound, 1.0)
        # compute error for false detection (FP)
        score_upper_error = K.clip(y_pred - score_upper_bound_target, 0.0, 1.0)
        # compute error for detection (TP)
        score_lower_error = K.clip(y_true - y_pred, 0.0, 1.0)

        score_upper_mse = K.mean(K.square(score_upper_error))
        score_lower_mse = K.sum(K.square(score_lower_error)) / (K.sum(y_true) + K.epsilon())
        score_mse = score_tp_weight * score_lower_mse + score_fp_weight * score_upper_mse

        return score_mse

    def size_loss(y_true, y_pred):
        # size_mask = K.cast(K.greater(y_true, 0.0), K.floatx())
        size_mask = K.clip(y_true, 0.0, 1.0)
        size_pred_masked = size_mask * y_pred
        # size_se = K.square(K.clip(y_true - y_pred, 0.0, 10000.0))
        # print(y_true, y_pred, size_pred_masked)
        size_se = K.square(K.square(y_true) - K.square(size_pred_masked))
        # return K.sum(size_mask)
        return K.sum(size_se) / (K.sum(size_mask) + K.epsilon())
        # return K.max(size_se)
        # return K.sum(size_mask)
    
    return {"Score": score_loss, "Size": size_loss}


def raw_output_loss(score_tp_weight: float = 1.0, score_fp_weight: float = 0.5, gaussian_diameter: int = 31,
                    gaussian_height: int = 5, score_min_bound=0.0):

    g = gaussian_kernel_3d(gaussian_diameter, gaussian_height)
    g /= g.max()
    g = g.reshape((gaussian_diameter, gaussian_diameter, gaussian_height, 1, 1))
    g = K.constant(g, dtype=np.float32)

    def raw_loss(y_true, y_pred):
        score_upper_bound_target = K.conv3d(y_true, g, padding="same")
        score_upper_bound_target = K.clip(score_upper_bound_target, score_min_bound, 1.0)
        # compute error for false detection (FP)
        score_upper_error = K.clip(y_pred - score_upper_bound_target, 0.0, 1.0)
        # compute error for detection (TP)
        score_lower_error = K.clip(y_true - y_pred, 0.0, 1.0)

        score_upper_mse = K.mean(K.square(score_upper_error))
        score_lower_mse = K.sum(K.square(score_lower_error)) / (K.sum(y_true) + K.epsilon())
        score_mse = score_tp_weight * score_lower_mse + score_fp_weight * score_upper_mse

        return score_mse

    return raw_loss
