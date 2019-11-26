import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
    Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf

import math

# from keras import backend as K
# from keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
#     Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape
# from keras.models import Model
# from keras.activations import softmax
# from keras.applications.mobilenet_v2 import MobileNetV2


# from utils import *

angle_count = 4
start_angle = math.pi / 4.0
edge_kernel_size = 3
line_detection_filter_size = 3
bounded_line_detection_filter_size = 5
square_detection_square_size_count = 2
square_detection_min_square_size = 3
square_detection_max_square_size = 4
square_detection_kernel_size = 3

class_count = 1  # excluding background (doesn't properly update hand crafted weights)


# detection_filter_filter_size = 5
# detection_filter_dmz_size = 1
# detection_filter_penalty = -0.1
# score_filter_size = 5
# size_filter_size = 5
# scale_initial_value = 1.0


def load_network(size_value, dropout_rate: float = 0.1, dropout_strategy: str = "all",
                 layers_filters: tuple = (16, 16, 24, 32), expansions: tuple = (1, 6, 6), print_summary=True,
                 use_resnet=False, use_mobile_net=False, use_additional_output=False):
    height, width = size_value

    ####################################
    # Custome Backbone
    ####################################

    squares = []
    prediction_shapes = []
    sizes = [6, 10, 15, 24, 42, 78]  # optimised for curve signs
    sizes = [int(s / 220 * height) for s in sizes]  # for smaller input size (than 220, 400)
    alpha = 1.0

    if not use_additional_output:
        sizes = sizes[:5]

    dropout_all = dropout_rate if dropout_strategy == "all" else None
    dropout_end = dropout_rate if dropout_strategy != "all" else None

    first_layer_kernel = 3

    f1, f2, f3, f4 = layers_filters
    e1, e2, e3 = expansions

    input_layer = Input(shape=(height, width, 3), name='input')
    x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(tf.keras.backend, input_layer, first_layer_kernel),
                                      name='Conv1_pad')(input_layer)
    x = Conv2D(filters=f1, kernel_size=first_layer_kernel, strides=2, activation=None, padding='valid',
               kernel_regularizer=l2(0.01),
               use_bias=False, name="Conv1")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.99, name="Conv1_BN")(x)
    x = tf.keras.layers.ReLU(6., name="Conv1_relu")(x)
    if dropout_all is not None:
        x = Dropout(dropout_all, name='Conv1_dropout')(x)
    x = _inverted_res_block(x, filters=f2, alpha=alpha, stride=1, expansion=e1, block_id=0, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    x = _inverted_res_block(x, filters=f3, alpha=alpha, stride=2, expansion=e2, block_id=1, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    x = _inverted_res_block(x, filters=f3, alpha=alpha, stride=1, expansion=e2, block_id=2, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    x = _inverted_res_block(x, filters=f3, alpha=alpha, stride=1, expansion=e2, block_id=3, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    if dropout_end is not None:
        x = Dropout(dropout_end, name='part1_dropout')(x)

    if use_additional_output:
        # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=10,
        #                           force_output_filter_count=True)
        out = Conv2D(filters=class_count + 1,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     activation='linear',
                     use_bias=True,
                     kernel_regularizer=l2(0.01),
                     name="output_0")(x)
        out = Softmax(axis=3, name="output_0_softmax")(out)
        squares.append(out)
        prediction_shapes.append(np.array(out.shape[1:3]))
    # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=10,
    #                           force_output_filter_count=True)
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01),
                 name="output_1")(x)
    out = Softmax(axis=3, name="output_1_softmax")(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=11,
    #                           force_output_filter_count=True)
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01),
                 name="output_2")(x)
    out = Softmax(axis=3, name="output_2_softmax")(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    x = _inverted_res_block(x, filters=f4, alpha=alpha, stride=2, expansion=e3, block_id=4, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    x = _inverted_res_block(x, filters=f4, alpha=alpha, stride=1, expansion=e3, block_id=5, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    x = _inverted_res_block(x, filters=f4, alpha=alpha, stride=1, expansion=e3, block_id=6, dropout_rate=dropout_all, use_resnet=use_resnet, inverted_bottle_neck=use_mobile_net)
    if dropout_end is not None:
        x = Dropout(dropout_end, name='part2_dropout')(x)

    # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=12,
    #                           force_output_filter_count=True)
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01),
                 name="output_3")(x)
    out = Softmax(axis=3, name="output_3_softmax")(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))
    # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=13,
    #                           force_output_filter_count=True)
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01),
                 name="output_4")(x)
    out = Softmax(axis=3, name="output_4_softmax")(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))
    # out = _inverted_res_block(x, filters=class_count + 1, alpha=alpha, stride=1, expansion=1, block_id=14,
    #                           force_output_filter_count=True)
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01),
                 name="output_5")(x)
    out = Softmax(axis=3, name="output_5_softmax")(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    print("Pyramid setup to track sizes: {}".format(sizes))
    print("Sizes on a 1080p video: {}".format([int(s / height * 1080) for s in sizes]))
    print("Pyramid prediction shapes are: {}".format(prediction_shapes))

    flatten_squares = [Reshape(target_shape=(x.shape[1] * x.shape[2], x.shape[3]), name="Flatten{}".format(i))(x)
                       for i, x in enumerate(squares)]

    prediction = Concatenate(axis=1, name="concatenate_final")(flatten_squares)

    model = Model(inputs=input_layer, outputs=prediction)

    if print_summary:
        model.summary()
    return model, sizes, prediction_shapes


"""
Code from keras_application MobileNet v2: 
https://github.com/keras-team/keras-applications/blob/71acdcd98088501247f4b514b7cbbdf8182a05a4/keras_applications/mobilenet_v2.py#L425
"""


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, force_output_filter_count: bool = False,
                        use_resnet: bool = False, dropout_rate: float = None, inverted_bottle_neck=False):
    if inverted_bottle_neck:
        return mobilenet_inverted_res_block(inputs, expansion, stride, alpha, filters, block_id,
                                            force_output_filter_count)
    else:
        return my_block(inputs, stride, filters, block_id, use_resnet, dropout_rate)


def my_block(inputs, stride, filters, block_id, use_resnet: bool = False, dropout_rate: float = None):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters=filters,
               kernel_size=3,
               strides=stride,
               padding='same',
               use_bias=False,
               activation=None,
               name='{}conv'.format(prefix),
               kernel_regularizer=l2(0.01))(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           epsilon=1e-3,
                                           momentum=0.99,
                                           name=prefix + 'BN')(x)
    x = tf.keras.layers.ReLU(6., name='{}relu'.format(prefix))(x)
    if use_resnet and tf.keras.backend.int_shape(inputs)[-1] == filters and stride == 1:
        x = tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
    if dropout_rate is not None:
        x = Dropout(dropout_rate, name=prefix + 'dropout')(x)
    return x


def mobilenet_inverted_res_block(inputs, expansion, stride, alpha, filters, block_id,
                                 force_output_filter_count: bool = False):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

    in_channels = tf.keras.backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    if not force_output_filter_count:
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    else:
        pointwise_filters = pointwise_conv_filters
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = tf.keras.layers.Conv2D(expansion * in_channels,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                               epsilon=1e-3,
                                               momentum=0.999,
                                               name=prefix + 'expand_BN')(x)
        x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(tf.keras.backend, x, 3),
                                          name=prefix + 'pad')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                        strides=stride,
                                        activation=None,
                                        use_bias=False,
                                        padding='same' if stride == 1 else 'valid',
                                        name=prefix + 'depthwise')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           epsilon=1e-3,
                                           momentum=0.999,
                                           name=prefix + 'depthwise_BN')(x)

    x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(pointwise_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               activation=None,
                               name=prefix + 'project')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           epsilon=1e-3,
                                           momentum=0.999,
                                           name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
    return x
