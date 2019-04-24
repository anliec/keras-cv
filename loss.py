from keras import backend as K

from utils import gaussian_kernel


def detection_loss(gaussian_area: int = 31, score_tp_weight: float = 0.9, score_fp_weight: float = 0.5,
                   score_weight: float = 0.5, size_weight: float = 0.5):
    def custom_loss(y_true, y_pred):
        y_pred_score, y_pred_size = y_pred
        y_true_score, y_true_size = y_true
    
        g = gaussian_kernel(gaussian_area)
        g /= g.max()
    
        score_upper_bound_target = K.conv2d(y_true_score, g)
        score_upper_error = K.clip(K.sum(y_pred_score, -score_upper_bound_target), 0.0, 1.0)
    
        score_lower_error = K.clip(K.sum(-y_pred_score, y_true_score), 0.0, 1.0)
    
        score_upper_mse = K.mean(K.square(score_upper_error))
        score_lower_mse = K.mean(K.square(score_lower_error))
        score_mse = score_tp_weight * score_lower_mse + score_fp_weight * score_upper_mse
    
        size_mask = K.greater(y_true_size, 0.0)
        size_pred_masked = size_mask * y_pred_size
        size_error = K.sum(-size_pred_masked, y_true_size)
        size_mse = K.mean(K.square(size_error))
        
        return score_mse * score_weight + size_mse * size_weight
    
    return custom_loss




