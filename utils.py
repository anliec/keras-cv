import math
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Layer
import scipy.stats as st
from tensorflow.python.keras import backend as K
from functools import lru_cache


def create_edge_kernel(left_color, right_color, angle, kernel_size=7, dmz_size=0.0) -> np.ndarray:
    angle = angle % (2 * math.pi)
    if math.pi / 2 < angle <= (3 * math.pi) / 2:
        return create_edge_kernel(right_color, left_color, angle - math.pi, kernel_size, dmz_size)
    assert kernel_size % 2 == 1
    assert len(right_color) == len(left_color)
    center = kernel_size // 2
    kernel = np.zeros(shape=(kernel_size,
                             kernel_size,
                             len(right_color)),
                      dtype=np.float32)
    ratio = math.tan(angle)
    dmz = abs(ratio * dmz_size)
    for y, vy in enumerate(kernel):
        for x, v in enumerate(vy):
            cur_x = x - center
            cur_y = y - center
            if ratio * cur_x + dmz < cur_y - dmz_size:
                v[:] = right_color[:]
            elif ratio * cur_x - dmz > cur_y + dmz_size:
                v[:] = left_color[:]
    return kernel / np.abs(kernel.sum())


def get_edge_kernel_weights(input_filters: int, kernel_size: int, filters_count: int, start_angle: float = 0.0):
    angle_increment = math.pi * 2 / filters_count
    layer_kernel = np.zeros(shape=(kernel_size, kernel_size, input_filters, filters_count))
    for i in range(filters_count):
        layer_kernel[:, :, :, i] = create_edge_kernel([1, 1, -2],
                                                      [-1, -1, -1],
                                                      i * angle_increment + start_angle,
                                                      kernel_size,
                                                      0.3)
    layer_bias = np.zeros(shape=filters_count)
    return layer_kernel, layer_bias


def get_edge_layer_and_weights(input_filters: int, kernel_size: int, filters_count: int, start_angle: float = 0.0,
                               padding='same', activation='relu'):
    layer = Conv2D(filters_count,
                   (kernel_size, kernel_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="EdgeDetector")
    kernel, bias = get_edge_kernel_weights(input_filters, kernel_size, filters_count, start_angle)
    return layer, kernel, bias


def bounded_line_detection_filter(filter_size, line_length, line_offset, line_angle) -> np.ndarray:
    filter_matrix = np.zeros(shape=(filter_size, filter_size), dtype=np.float32)
    center = filter_size // 2
    x_lenght = abs(round(line_length * math.cos(line_angle)))
    line_middle_x = round(-line_offset * math.sin(line_angle))
    if x_lenght <= 1:
        p1_max = min(filter_size, center + line_length + 1)
        p2_max = min(filter_size, center + line_length//2 + 1)
        p3_max = max(0, center - line_length//2)
        p3_min = max(0, center - line_length)
        x_pos = max(0, min(filter_size - 1, center + line_middle_x))
        filter_matrix[p2_max:p1_max, x_pos] = -1.
        filter_matrix[p3_max:p2_max, x_pos] = 1.
        filter_matrix[p3_min:p3_max, x_pos] = -1.
        return filter_matrix
    a = math.tan(line_angle)
    b = line_offset * math.cos(line_angle) - a * line_middle_x

    def f(x: int):
        point_list_x = []
        point_list_y = []
        if -center <= x <= center:
            v_start = round(a * (x-0.5) + b)
            v_end = round(a * (x+0.5) + b)
            v_min = min(v_start, v_end)
            v_max = max(v_start, v_end)
            if v_min == v_max:
                v_max += 1
            for v in range(max(-center, v_min), min(center + 1, v_max)):
                point_list_x.append(x + center)
                point_list_y.append(v + center)
        return point_list_y, point_list_x

    negative_points_y, negative_points_x = [], []
    positive_points_y, positive_points_x = [], []
    for x in range(line_middle_x - x_lenght,
                   line_middle_x - x_lenght//2):
        lx, ly = f(x)
        negative_points_x += lx
        negative_points_y += ly
    for x in range(line_middle_x - x_lenght//2,
                   line_middle_x + 1 + x_lenght//2):
        lx, ly = f(x)
        positive_points_x += lx
        positive_points_y += ly
    for x in range(line_middle_x + 1 + x_lenght//2,
                   line_middle_x + 1 + x_lenght):
        lx, ly = f(x)
        negative_points_x += lx
        negative_points_y += ly
    try:
        negative_weight = -1.0 / len(negative_points_y)
    except ZeroDivisionError as e:
        print(e)
        negative_weight = -1.0
    positive_weight = 1.0 / len(positive_points_y)
    filter_matrix[(negative_points_y, negative_points_x)] = negative_weight
    filter_matrix[(positive_points_y, positive_points_x)] = positive_weight
    return filter_matrix


def get_square_detection_weights(input_filter_count: int, filter_count: int, min_square_size: int,
                                 max_square_size: int, kernel_size: int, first_input_filter_index: int = 1,
                                 input_filter_index_increment: int = 2, start_angle: float = 0.0):
    kernel = np.zeros(shape=(kernel_size, kernel_size, input_filter_count, filter_count))
    bias = np.zeros(shape=filter_count)

    for f in range(filter_count):
        square_size = f * (max_square_size - min_square_size) + min_square_size
        for i in range(first_input_filter_index, input_filter_count, input_filter_index_increment):
            angle = i * math.pi * 2 / input_filter_count + start_angle
            kernel[:, :, i, f] = bounded_line_detection_filter(kernel_size,
                                                               square_size,
                                                               square_size // 2,
                                                               angle)
    return kernel, bias


def get_square_detection_layer_and_weights(input_filter_count: int, filter_count: int, min_square_size: int,
                                           max_square_size: int, kernel_size: int, first_input_filter_index: int = 1,
                                           input_filter_index_increment: int = 2, start_angle: float = 0.0,
                                           padding='same', activation='relu'):
    layer = Conv2D(filter_count,
                   (kernel_size, kernel_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="SquareDetector")
    kernel, bias = get_square_detection_weights(input_filter_count, filter_count, min_square_size, max_square_size,
                                                kernel_size, first_input_filter_index, input_filter_index_increment,
                                                start_angle)
    return layer, kernel, bias


def small_filter_line_detector(filter_size: int, line_angle: float):
    filter_matrix = np.zeros(shape=(filter_size, filter_size), dtype=np.float32)
    center = filter_size // 2
    a = math.tan(line_angle)

    def f(x: int):
        point_list_x = []
        point_list_y = []
        if -center <= x <= center:
            v_start = round(a * (x-0.5))
            v_end = round(a * (x+0.5))
            v_min = min(v_start, v_end)
            v_max = max(v_start, v_end)
            if v_max - v_min == 1 and v_max == round(a * x):
                v_min += 1
                v_max += 1
            elif v_min == v_max:
                v_max += 1
            for v in range(max(-center, v_min), min(center + 1, v_max)):
                point_list_x.append(x + center)
                point_list_y.append(v + center)
        return point_list_y, point_list_x

    for x in range(0, filter_size):
        filter_matrix[f(x - center)] = 1.
    return filter_matrix / filter_matrix.sum()


def get_line_detection_weights(filter_count: int, angle_increment: float, filter_size: int = 5,
                               start_angle: float = 0.0):
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             filter_count,
                             filter_count))
    bias = np.zeros(shape=filter_count)

    for f in range(filter_count):
        angle = f * angle_increment + start_angle
        kernel[:, :, f, f] = small_filter_line_detector(filter_size,
                                                        angle)

    return kernel, bias


def get_line_detection_layer_and_weights(filter_count: int, angle_increment: float, filter_size: int = 5,
                                         start_angle: float = 0.0, padding='same', activation='relu'):
    layer = Conv2D(filter_count,
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="LineDetector")
    kernel, bias = get_line_detection_weights(filter_count, angle_increment, filter_size, start_angle)
    return layer, kernel, bias


def get_bounded_line_detection_weights(filter_count: int, angle_increment: float, line_lengths: list,
                                       filter_size: int = 5, start_angle: float = 0.0):
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             filter_count,
                             filter_count * len(line_lengths)))
    bias = np.zeros(shape=filter_count * len(line_lengths))

    for i, l in enumerate(line_lengths):
        out_offset = i * filter_count
        for f in range(filter_count):
            angle = f * angle_increment + start_angle
            kernel[:, :, f, f + out_offset] = bounded_line_detection_filter(filter_size=filter_size,
                                                                            line_length=l,
                                                                            line_angle=angle,
                                                                            line_offset=0)
    return kernel, bias


def get_bounded_line_detection_layer_and_weights(filter_count: int, angle_increment: float, line_lengths: list,
                                                 filter_size: int = 5, start_angle: float = 0.0, padding='same',
                                                 activation='relu'):
    layer = Conv2D(filter_count * len(line_lengths),
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="BoundedLineDetector")
    kernel, bias = get_bounded_line_detection_weights(filter_count, angle_increment, line_lengths, filter_size,
                                                      start_angle)
    return layer, kernel, bias


def get_square_detection_weights_from_bounded_lines(filter_count: int, angle_increment: float, line_lengths: list,
                                                    filter_size: int = 5, start_angle: float = 0.0,
                                                    pooling_before: int = 1):
    assert angle_increment == math.pi / 2.0
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             filter_count,
                             len(line_lengths)))
    bias = np.zeros(shape=len(line_lengths))

    for i, length in enumerate(line_lengths):
        for j in range(4):
            a = (math.pi / 4.0) + start_angle + j * (math.pi / 2.0)
            d = length / 2.0
            # convert polar to cartesian
            x = math.cos(a) * d
            y = math.sin(a) * d
            # apply pooling scaling factor
            x /= pooling_before
            y /= pooling_before
            # center coordinates of filter
            x += (filter_size - 1) // 2
            y += (filter_size - 1) // 2
            # get integer coordinates
            x = int(x)
            y = int(y)
            # ensure coordinates on filter
            x = min(filter_size - 1, max(0, x))
            y = min(filter_size - 1, max(0, y))
            # set kernel value
            kernel[x, y, i * len(line_lengths) + j, i] = 1.0
    return kernel, bias


def get_square_detection_layer_and_weights_from_bounded_lines(filter_count: int, angle_increment: float,
                                                              line_lengths: list, filter_size: int = 5,
                                                              start_angle: float = 0.0, pooling_before: int = 1,
                                                              padding='same', activation='relu'):
    layer = Conv2D(len(line_lengths),
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="SquareByLineDetector")
    kernel, bias = get_square_detection_weights_from_bounded_lines(filter_count, angle_increment, line_lengths,
                                                                   filter_size, start_angle, pooling_before)
    return layer, kernel, bias


def get_detection_filter_layer_and_weights(filter_count: int, dmz_size: int = 1, other_penalty: float = -1,
                                           filter_size: int = 5, padding='same', activation='relu'):
    layer = Conv2D(filter_count,
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="DetectionFiltering")
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             filter_count,
                             filter_count))
    bias = np.zeros(shape=filter_count)
    gaussian = gaussian_kernel(filter_size)
    for f_in in range(filter_count):
        for f_out in range(filter_count):
            if f_in == f_out:
                w = gaussian
            elif f_in + dmz_size < f_out or f_in - dmz_size > f_out:
                w = other_penalty * gaussian
            else:  # DMZ case
                w = np.zeros(shape=(filter_size, filter_size), dtype=np.float32)
            kernel[:, :, f_in, f_out] = w
    return layer, kernel, bias


def get_score_layer_and_weights(input_filter: int, filter_size: int = 5, padding='same', activation='relu'):
    layer = Conv2D(1,
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="Score")
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             input_filter,
                             1))
    bias = np.zeros(shape=1)
    gaussian = gaussian_kernel(filter_size)
    for f in range(input_filter):
        kernel[:, :, f, 0] = gaussian
    return layer, kernel, bias


def get_size_layer_and_weights(input_filter: int, filter_size: int = 5, first_filter_size=5, filter_size_factor=2,
                               padding='same', activation='relu'):
    layer = Conv2D(1,
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation,
                   name="Size")
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             input_filter,
                             1))
    bias = np.zeros(shape=1)
    gaussian = gaussian_kernel(filter_size)
    for f in range(input_filter):
        kernel[:, :, f, 0] = gaussian * first_filter_size * math.pow(filter_size_factor, input_filter - f - 1)
    return layer, kernel, bias


@lru_cache(maxsize=20)
def gaussian_kernel(kernel_size: int) -> np.ndarray:
    """Returns a 2D Gaussian kernel."""

    lim = kernel_size // 2 + (kernel_size % 2) / 2
    x = np.linspace(-lim, lim, kernel_size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


@lru_cache(maxsize=20)
def gaussian_kernel_3d(kernel_size: int, kernel_height: int) -> np.ndarray:
    """Returns a 2D Gaussian kernel."""
    kern2d = gaussian_kernel(kernel_size)
    lim = kernel_height // 2 + (kernel_height % 2) / 2
    x = np.linspace(-lim, lim, kernel_height + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern3d = np.zeros(shape=(kernel_size, kernel_size, kernel_height))
    for i, line in enumerate(kern2d):
        kern3d[i, :, :] = np.outer(line, kern1d)
    return kern3d/kern3d.sum()


def normalisation(x, axis=-1):
    """normalisation activation function.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the normalization is applied.

    # Returns
        Tensor, output of normalisation transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply normalisation to a tensor that is 1D')
    elif ndim > 1:
        p = K.softplus(x)
        s = K.sum(p, axis=axis, keepdims=True)
        return p / s
    else:
        raise ValueError('Cannot apply normalisation to a tensor that is 1D. '
                         'Received input: %s' % x)


class Normalisation(Layer):
    """Normalisation activation function.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        axis: Integer, axis along which the Normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Normalisation, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return normalisation(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Normalisation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



