import numpy as np
from keras import backend as K
from keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
    Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape
from keras.models import Model
from keras.activations import softmax
from utils import *

angle_count = 4
start_angle = math.pi / 4.0
edge_kernel_size = 3
line_detection_filter_size = 3
square_detection_square_size_count = 2
square_detection_min_square_size = 3
square_detection_max_square_size = 4
square_detection_kernel_size = 11
# detection_filter_filter_size = 5
# detection_filter_dmz_size = 1
# detection_filter_penalty = -0.1
# score_filter_size = 5
# size_filter_size = 5
# scale_initial_value = 1.0


def load_network(size_value, random_init: bool = False, first_pyramid_output: int = 2, pyramid_depth: int = 7):
    assert len(size_value) == 2
    k = math.pow(2, pyramid_depth - 1)
    for i, v in enumerate(size_value):
        v -= (v - edge_kernel_size + 1) % k
        size_value[i] = int(v)
    print("Reshaped input size: {}".format(size_value))
    height, width = size_value

    sizes = []
    square_size_increment = ((square_detection_max_square_size - square_detection_min_square_size)
                             / square_detection_square_size_count)
    for pyramid_level in range(first_pyramid_output, pyramid_depth):
        factor = math.pow(2, pyramid_level)
        for square_size_index in range(square_detection_square_size_count):
            square_size = square_detection_min_square_size + square_size_increment * square_size_index
            sizes.append(square_size * factor)

    print("Pyramid setup to track sizes: {}".format(sizes))

    # create layers
    input_layer = Input(shape=(height, width, 3))
    edge_layer, edge_weights, edge_bias = get_edge_layer_and_weights(
        input_filters=3,
        kernel_size=edge_kernel_size,
        filters_count=angle_count,
        start_angle=start_angle,
        padding='valid'
    )
    line_layer, line_weights, line_bias = get_line_detection_layer_and_weights(
        filter_count=angle_count,
        angle_increment=math.pi * 2 / angle_count,
        filter_size=line_detection_filter_size,
        start_angle=start_angle,
        padding='same'
    )
    square_layer, square_weights, square_bias = get_square_detection_layer_and_weights(
        input_filter_count=angle_count,
        filter_count=square_detection_square_size_count,
        min_square_size=square_detection_min_square_size,
        max_square_size=square_detection_max_square_size,
        kernel_size=square_detection_kernel_size,
        start_angle=start_angle,
        padding='same'
    )
    max_pool_layer = MaxPool2D(pool_size=(2, 2), padding='same')
    # filter_layer, filter_weights, filter_bias = get_detection_filter_layer_and_weights(
    #     filter_count=pyramid_depth - first_pyramid_output,
    #     dmz_size=detection_filter_dmz_size,
    #     other_penalty=detection_filter_penalty,
    #     filter_size=detection_filter_filter_size
    # )
    # score_layer, score_weights, score_bias = get_score_layer_and_weights(
    #     input_filter=pyramid_depth - first_pyramid_output,
    #     filter_size=score_filter_size,
    #     activation='tanh'
    # )
    # size_layer, size_weights, size_bias = get_size_layer_and_weights(
    #     input_filter=pyramid_depth - first_pyramid_output,
    #     filter_size=size_filter_size,
    #     first_filter_size=line_detection_filter_size,
    #     filter_size_factor=2
    # )

    # create model
    edges = edge_layer(input_layer)
    line = line_layer(edges)
    # pool = max_pool_layer(line)
    pyramid = [line, ]
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
    s = [s.value for s in concat.shape[1:]]
    concat = Reshape(target_shape=s + [1])(concat)

    # filtered = filter_layer(concat)

    # score = score_layer(filtered)

    # size_score = Softmax(axis=3, name="SoftmaxNorm")(filtered)
    # size_score = Normalisation(axis=3, name="Normalisation")(filtered)
    # size_score = BatchNormalization(axis=3)(filtered)
    # size_value = size_layer(filtered)

    # scale_factor = tf.Variable(scale_initial_value, name='scale_factor', dtype=np.float32)
    # input_shape = Lambda(lambda x: x * scale_factor, name="Size")(input_shape)

    # model = Model(inputs=input_layer, outputs=[score, size_value])
    model = Model(inputs=input_layer, outputs=concat)
    # model.compile("SGD", loss='mse')

    # set weights
    if not random_init:
        print("Loading preset weights")
        model.get_layer("EdgeDetector").set_weights((edge_weights, edge_bias))
        model.get_layer("LineDetector").set_weights((line_weights, line_bias))
        model.get_layer("SquareDetector").set_weights((square_weights, square_bias))
        # model.get_layer("DetectionFiltering").set_weights((filter_weights, filter_bias))
        # model.get_layer("Score").set_weights((score_weights, score_bias))
        # model.get_layer("Size").set_weights((size_weights, score_bias))
    else:
        print("Keeping random weights")
    model.summary()
    return model, sizes
