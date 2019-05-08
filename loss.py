from keras import backend as K
import numpy as np

from utils import gaussian_kernel


def detection_loss(gaussian_diameter: int = 31, score_tp_weight: float = 0.9, score_fp_weight: float = 0.5,
                   score_weight: float = 0.5, size_weight: float = 0.5):
    g = np.zeros(shape=(gaussian_diameter, gaussian_diameter, 1, 1), dtype=np.float32)
    g[:, :, 0, 0] = gaussian_kernel(gaussian_diameter)
    g /= g.max()
    g = K.constant(g, dtype=np.float32)

    def score_loss(y_true, y_pred):
        score_upper_bound_target = K.conv2d(y_true, g, padding="same")
        # compute error for false detection (FP)
        score_upper_error = K.clip(y_pred - score_upper_bound_target, 0.0, 1.0)
        # compute error for detection (TP)
        score_lower_error = K.clip(y_true - y_pred, 0.0, 1.0)

        score_upper_mse = K.mean(K.square(score_upper_error))
        score_lower_mse = K.sum(K.square(score_lower_error)) / (K.sum(y_true) + 1)
        score_mse = score_tp_weight * score_lower_mse + score_fp_weight * score_upper_mse

        return score_mse

    def size_loss(y_true, y_pred):
        # size_mask = K.cast(K.greater(y_true, 0.0), K.floatx())
        # size_pred_masked = size_mask * y_pred
        size_error = K.clip(y_true - y_pred, 0.0, 10000.0)
        size_mse = K.square(size_error)
        
        return K.max(size_mse)
    
    return {"Score": score_loss, "Size": size_loss}




