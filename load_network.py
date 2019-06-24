import numpy as np
from keras import backend as K
from keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
    Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax
from keras.models import Model
from keras.activations import softmax
from utils import *

angle_count = 8
edge_kernel_size = 7
line_detection_filter_size = 5
square_detection_square_size_count = 1
square_detection_min_square_size = 5
square_detection_max_square_size = 5
square_detection_kernel_size = 13
detection_filter_filter_size = 5
detection_filter_dmz_size = 1
detection_filter_penalty = -0.1
score_filter_size = 5
size_filter_size = 5
first_pyramid_output = 2
pyramid_depth = 7
scale_initial_value = 1.0


def load_network(input_shape):
    assert len(input_shape) == 2
    k = math.pow(2, pyramid_depth)
    for i, v in enumerate(input_shape):
        v -= (v - edge_kernel_size + 1) % k
        input_shape[i] = int(v)
    print("Reshaped input size: {}".format(input_shape))
    height, width = input_shape

    # create layers
    input_layer = Input(shape=(height, width, 3))
    edge_layer, edge_weights, edge_bias = get_edge_layer_and_weights(
        input_filters=3,
        kernel_size=edge_kernel_size,
        filters_count=angle_count,
        padding='valid'
    )
    line_layer, line_weights, line_bias = get_line_detection_layer_and_weights(
        filter_count=angle_count,
        angle_increment=math.pi * 2 / angle_count,
        filter_size=line_detection_filter_size
    )
    square_layer, square_weights, square_bias = get_square_detection_layer_and_weights(
        input_filter_count=angle_count,
        filter_count=square_detection_square_size_count,
        min_square_size=square_detection_min_square_size,
        max_square_size=square_detection_max_square_size,
        kernel_size=square_detection_kernel_size
    )
    max_pool_layer = MaxPool2D(pool_size=(2, 2), padding='same')
    filter_layer, filter_weights, filter_bias = get_detection_filter_layer_and_weights(
        filter_count=pyramid_depth - first_pyramid_output,
        dmz_size=detection_filter_dmz_size,
        other_penalty=detection_filter_penalty,
        filter_size=detection_filter_filter_size
    )
    score_layer, score_weights, score_bias = get_score_layer_and_weights(
        input_filter=pyramid_depth - first_pyramid_output,
        filter_size=score_filter_size,
        activation='tanh'
    )
    size_layer, size_weights, size_bias = get_size_layer_and_weights(
        input_filter=pyramid_depth - first_pyramid_output,
        filter_size=size_filter_size,
        first_filter_size=line_detection_filter_size,
        filter_size_factor=2
    )

    # create model
    edges = edge_layer(input_layer)
    line = line_layer(edges)
    pool = max_pool_layer(line)
    pyramid = [pool, ]
    for i in range(1, pyramid_depth):
        line = line_layer(pyramid[-1])
        pool = max_pool_layer(line)
        pyramid.append(pool)

    squares = [square_layer(l) for l in pyramid[first_pyramid_output:]]

    upsamplings = []
    for i, s in enumerate(squares):
        n = math.pow(2, i)
        up = UpSampling2D(size=(n, n), interpolation='bilinear')(s)
        upsamplings.append(up)

    concat = Concatenate(axis=3)(upsamplings)

    filtered = filter_layer(concat)

    score = score_layer(filtered)

    size_score = Softmax(axis=3, name="SoftmaxNorm")(filtered)
    # size_score = Normalisation(axis=-1, name="Normalisation")(filtered)
    # size_score = BatchNormalization(axis=-1)(filtered)
    input_shape = size_layer(size_score)

    # scale_factor = tf.Variable(scale_initial_value, name='scale_factor', dtype=np.float32)
    # input_shape = Lambda(lambda x: x * scale_factor, name="Size")(input_shape)

    model = Model(inputs=input_layer, outputs=[score, input_shape])
    # model.compile("SGD", loss='mse')

    # set weights
    model.get_layer("EdgeDetector").set_weights((edge_weights, edge_bias))
    model.get_layer("LineDetector").set_weights((line_weights, line_bias))
    model.get_layer("SquareDetector").set_weights((square_weights, square_bias))
    model.get_layer("DetectionFiltering").set_weights((filter_weights, filter_bias))
    model.get_layer("Score").set_weights((score_weights, score_bias))
    model.get_layer("Size").set_weights((size_weights, score_bias))

    model.summary()
    return model