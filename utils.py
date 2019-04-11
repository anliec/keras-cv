import math
import numpy as np
from keras.layers import Conv2D, UpSampling2D


def create_edge_kernel(left_color, right_color, angle, kernel_size=7, dmz_size=0.0) -> np.ndarray:
    angle = angle % (2 * math.pi)
    if math.pi / 2 < angle <= (3 * math.pi) / 2:
        return create_edge_kernel(right_color, left_color, angle - math.pi, kernel_size, dmz_size)
    assert kernel_size % 2 == 1
    assert len(right_color) == len(left_color)
    center = kernel_size // 2
    kernel = np.zeros(shape=(kernel_size, kernel_size, len(right_color)), dtype=np.float32)
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
    return kernel


def get_edge_kernel_weights(input_filters: int, kernel_size: int, filters_count: int):
    angle_increment = math.pi * 2 / filters_count
    layer_kernel = np.zeros(shape=(kernel_size, kernel_size, input_filters, filters_count))
    for i in range(filters_count):
        layer_kernel[:, :, :, i] = create_edge_kernel([1, 1, -2],
                                                      [-1, -1, -1],
                                                      i * angle_increment,
                                                      kernel_size,
                                                      0.3
                                                      )
    layer_bias = np.zeros(shape=filters_count)
    return layer_kernel, layer_bias


def get_edge_layer_and_weights(input_filters: int, kernel_size: int, filters_count: int, padding='valid',
                               activation='relu'):
    layer = Conv2D(filters_count,
                   (kernel_size, kernel_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation)
    kernel, bias = get_edge_kernel_weights(input_filters, kernel_size, filters_count)
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

    def f(x:int):
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

    for x in range(line_middle_x - x_lenght,
                   line_middle_x - x_lenght//2):
        filter_matrix[f(x)] = -1.
    for x in range(line_middle_x - x_lenght//2,
                   line_middle_x + 1 + x_lenght//2):
        filter_matrix[f(x)] = 1.
    for x in range(line_middle_x + 1 + x_lenght//2,
                   line_middle_x + 1 + x_lenght):
        filter_matrix[f(x)] = -1.
    return filter_matrix


def get_square_detection_weights(input_filter_count: int, filter_count: int, min_square_size: int,
                                 max_square_size: int, kernel_size: int, first_input_filter_index: int = 1,
                                 input_filter_index_increment: int = 2):
    kernel = np.zeros(shape=(kernel_size, kernel_size, input_filter_count, filter_count))
    bias = np.zeros(shape=filter_count)

    for f in range(filter_count):
        square_size = f * (max_square_size - min_square_size) + min_square_size
        for i in range(first_input_filter_index, input_filter_count, input_filter_index_increment):
            angle = i * math.pi * 2 / input_filter_count
            kernel[:, :, i, f] = bounded_line_detection_filter(kernel_size,
                                                               square_size,
                                                               square_size // 2,
                                                               angle)
    return kernel, bias


def get_square_detection_layer_and_weights(input_filter_count: int, filter_count: int, min_square_size: int,
                                           max_square_size: int, kernel_size: int, first_input_filter_index: int = 1,
                                           input_filter_index_increment: int = 2, padding='valid',
                                           activation='relu'):
    layer = Conv2D(filter_count,
                   (kernel_size, kernel_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation)
    kernel, bias = get_square_detection_weights(input_filter_count, filter_count, min_square_size, max_square_size,
                                                kernel_size, first_input_filter_index, input_filter_index_increment)
    return layer, kernel, bias


def small_filter_line_detector(filter_size: int, line_angle: float):
    filter_matrix = np.zeros(shape=(filter_size, filter_size), dtype=np.float32)
    center = filter_size // 2
    a = math.tan(line_angle)

    def f(x:int):
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


def get_line_detection_weights(filter_count: int, angle_increment: int, filter_size: int = 5):
    kernel = np.zeros(shape=(filter_size,
                             filter_size,
                             filter_count,
                             filter_count))
    bias = np.zeros(shape=filter_count)

    for f in range(filter_count):
        angle = f * angle_increment
        kernel[:, :, f, f] = small_filter_line_detector(filter_size,
                                                        angle)

    return kernel, bias


def get_line_detection_layer_and_weights(filter_count: int, angle_increment: int, filter_size: int = 5,
                                         padding='valid', activation='relu'):
    layer = Conv2D(filter_count,
                   (filter_size, filter_size),
                   padding=padding,
                   kernel_initializer='normal',
                   use_bias=True,
                   activation=activation)
    kernel, bias = get_line_detection_weights(filter_count, angle_increment, filter_size)
    return layer, kernel, bias

