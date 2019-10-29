import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
    Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf

# from keras import backend as K
# from keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
#     Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape
# from keras.models import Model
# from keras.activations import softmax
# from keras.applications.mobilenet_v2 import MobileNetV2


from utils import *

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


def load_network(size_value, random_init: bool = False, first_pyramid_output: int = 2, pyramid_depth: int = 7,
                 add_noise: bool = False):
    height, width = size_value

    ####################################
    # Custome Backbone
    ####################################

    squares = []
    prediction_shapes = []
    sizes = [6, 10, 15, 24, 42]  # optimised for curve signs
    dropout_rate = 0.5
    alpha = 1.0

    # def relu6(t):
    #     return tf.keras.activations.relu(t, max_value=6.0, threshold=0.0)

    input_layer = Input(shape=(height, width, 3))
    x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(tf.keras.backend, input_layer, 3),
                                      name='Conv1_pad')(input_layer)
    x = Conv2D(filters=32, kernel_size=5, strides=2, activation=None, padding='valid', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6., name="Conv1_relu")(x)
    x = Dropout(dropout_rate)(x)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    x = Dropout(dropout_rate)(x)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = Dropout(dropout_rate)(x)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    x = Dropout(dropout_rate)(x)

    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01))(x)
    out = Softmax(axis=3)(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01))(x)
    out = Softmax(axis=3)(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = Dropout(dropout_rate)(x)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = Dropout(dropout_rate)(x)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    x = Dropout(dropout_rate)(x)

    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01))(x)
    out = Softmax(axis=3)(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01))(x)
    out = Softmax(axis=3)(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))
    out = Conv2D(filters=class_count + 1,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='linear',
                 use_bias=True,
                 kernel_regularizer=l2(0.01))(x)
    out = Softmax(axis=3)(out)
    squares.append(out)
    prediction_shapes.append(np.array(out.shape[1:3]))

    print("Pyramid setup to track sizes: {}".format(sizes))
    print("Sizes on a 1080p video: {}".format([int(s / height * 1080) for s in sizes]))
    print("Pyramid prediction shapes are: {}".format(prediction_shapes))

    flatten_squares = [Reshape(target_shape=(x.shape[1] * x.shape[2], x.shape[3]))(x) for x in squares]

    prediction = Concatenate(axis=1)(flatten_squares)

    model = Model(inputs=input_layer, outputs=prediction)

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


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

    in_channels = tf.keras.backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
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
