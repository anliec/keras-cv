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

    def relu6(t):
        return tf.keras.activations.relu(t, max_value=6.0, threshold=0.0)

    input_layer = Input(shape=(height, width, 3))
    x = input_layer
    x = Conv2D(filters=16, kernel_size=5, strides=2, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=16, kernel_size=3, strides=1, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
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

    x = Conv2D(filters=64, kernel_size=3, strides=2, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu6, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
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
