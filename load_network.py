import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, UpSampling2D, \
    Concatenate, Maximum, Add, Activation, Lambda, BatchNormalization, Softmax, Reshape
from tensorflow.keras.models import Model
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
    # assert len(size_value) == 2
    # k = math.pow(2, pyramid_depth)
    # for i, v in enumerate(size_value):
    #     v -= (v - edge_kernel_size + 1) % k
    #     size_value[i] = int(v)
    # print("Reshaped input size: {}".format(size_value))
    height, width = size_value
    size_value = np.array(size_value)

    sizes = []
    square_size_increment = ((square_detection_max_square_size - square_detection_min_square_size)
                             / square_detection_square_size_count)
    for pyramid_level in range(first_pyramid_output, pyramid_depth):
        factor = math.pow(2, pyramid_level)
        for square_size_index in range(square_detection_square_size_count):
            square_size = square_detection_min_square_size + square_size_increment * square_size_index
            sizes.append(square_size * factor)

    prediction_shapes = [((size_value - (edge_kernel_size - 1)) / (2**(pyramid_level + 1))).astype(np.int)
                         for pyramid_level in range(first_pyramid_output, pyramid_depth)]

    print("Pyramid setup to track sizes: {}".format(sizes))
    print("Sizes on a 1080p video: {}".format([int(s / height * 1080) for s in sizes]))
    print("Pyramid prediction shapes are: {}".format(prediction_shapes))

    # input_layer = Input(shape=(height, width, 3))

    mobile_netv2 = MobileNetV2(input_shape=(height, width, 3),
                               include_top=False,
                               weights='imagenet'
                               )

    mobile_netv2.summary()

    input_layer = mobile_netv2.inputs[0]

    squares = []
    prediction_shapes = []
    first_layer = 18 + 19 * 1
    for i in range(first_layer, first_layer + (19 * pyramid_level), 19):
        l = mobile_netv2.layers[i].output
        classification_layer = Conv2D(filters=class_count + 1,
                                      kernel_size=(square_detection_kernel_size, square_detection_kernel_size),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear',
                                      use_bias=True)
        x = classification_layer(l)
        x = Softmax(axis=3)(x)
        squares.append(x)
        prediction_shapes.append(np.array(x.shape[1:3]))

    # create layers
    # input_layer = Input(shape=(height, width, 3))
    # edge_layer, edge_weights, edge_bias = get_edge_layer_and_weights(
    #     input_filters=3,
    #     kernel_size=edge_kernel_size,
    #     filters_count=angle_count,
    #     start_angle=start_angle,
    #     padding='valid'
    # )
    # line_layer, line_weights, line_bias = get_line_detection_layer_and_weights(
    #     filter_count=angle_count,
    #     angle_increment=math.pi * 2 / angle_count,
    #     filter_size=line_detection_filter_size,
    #     start_angle=start_angle,
    #     padding='same'
    # )
    # mi, ma, c = square_detection_min_square_size, square_detection_max_square_size, square_detection_square_size_count
    # lines_lengths = [mi + (ma - mi) / (c - 1) * i for i in range(c)]
    # bline_layer, b_line_weights, b_line_bias = get_bounded_line_detection_layer_and_weights(
    #     filter_count=angle_count,
    #     angle_increment=math.pi * 2 / angle_count,
    #     filter_size=bounded_line_detection_filter_size,
    #     start_angle=start_angle,
    #     line_lengths=lines_lengths,
    #     padding='same'
    # )
    # bsquare_layer, bsquare_weights, bsquare_bias = get_square_detection_layer_and_weights_from_bounded_lines(
    #     filter_count=angle_count * square_detection_square_size_count,
    #     angle_increment=math.pi * 2 / angle_count,
    #     filter_size=square_detection_kernel_size,
    #     start_angle=start_angle,
    #     line_lengths=lines_lengths,
    #     pooling_before=2,
    #     padding='same',
    #     activation='sigmoid'
    # )

    # classification_layer = Conv2D(filters=class_count + 1,
    #                               kernel_size=(square_detection_kernel_size, square_detection_kernel_size),
    #                               strides=(1, 1),
    #                               padding='same',
    #                               activation='linear',
    #                               use_bias=True)

    # square_layer, square_weights, square_bias = get_square_detection_layer_and_weights(
    #     input_filter_count=angle_count,
    #     filter_count=square_detection_square_size_count,
    #     min_square_size=square_detection_min_square_size,
    #     max_square_size=square_detection_max_square_size,
    #     kernel_size=square_detection_kernel_size,
    #     start_angle=start_angle,
    #     padding='same'
    # )
    # max_pool_layer = MaxPool2D(pool_size=(2, 2), padding='same')
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
    # edges = edge_layer(input_layer)
    # line = line_layer(edges)
    # pool = max_pool_layer(line)
    # pyramid = [edges, ]
    # for i in range(1, pyramid_depth):
    #     line = line_layer(pyramid[-1])
    #     pool = MaxPool2D(pool_size=(2, 2), padding='same')(line)
    #     pyramid.append(pool)
    #
    # # squares = [square_layer(l) for l in pyramid[first_pyramid_output:]]
    # squares = []
    # for l in pyramid[first_pyramid_output:]:
    #     x = bline_layer(l)
    #     x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    #     x = classification_layer(x)
    #     x = Softmax(axis=3)(x)
    #     squares.append(x)

    # upsamplings = []
    # for i, s in enumerate(squares):
    #     n = int(math.pow(2, i))
    #     up = UpSampling2D(size=(n, n), interpolation='bilinear')(s)
    #     upsamplings.append(up)
    #
    # concat = Concatenate(axis=3)(upsamplings)
    # try:
    #     s = [s.value for s in concat.shape[1:]]
    # except AttributeError:
    #     s = concat.shape[1:]
    # concat = Reshape(target_shape=s + [1])(concat)

    flatten_squares = [Reshape(target_shape=(-1, class_count + 1))(x) for x in squares]

    prediction = Concatenate(axis=1)(flatten_squares)

    # filtered = filter_layer(concat)

    # score = score_layer(filtered)

    # size_score = Softmax(axis=3, name="SoftmaxNorm")(filtered)
    # size_score = Normalisation(axis=3, name="Normalisation")(filtered)
    # size_score = BatchNormalization(axis=3)(filtered)
    # size_value = size_layer(filtered)

    # scale_factor = tf.Variable(scale_initial_value, name='scale_factor', dtype=np.float32)
    # input_shape = Lambda(lambda x: x * scale_factor, name="Size")(input_shape)

    # model = Model(inputs=input_layer, outputs=[score, size_value])
    model = Model(inputs=input_layer, outputs=prediction)
    # model.compile("SGD", loss='mse')

    # set weights
    # if not random_init:
    #     print("Loading preset weights")
    #     model.get_layer("EdgeDetector").set_weights((edge_weights, edge_bias))
    #     model.get_layer("LineDetector").set_weights((line_weights, line_bias))
    #     # model.get_layer("SquareDetector").set_weights((square_weights, square_bias))
    #     model.get_layer("BoundedLineDetector").set_weights((b_line_weights, b_line_bias))
    #     # model.get_layer("SquareByLineDetector").set_weights((bsquare_weights, bsquare_bias))
    #     # model.get_layer("DetectionFiltering").set_weights((filter_weights, filter_bias))
    #     # model.get_layer("Score").set_weights((score_weights, score_bias))
    #     # model.get_layer("Size").set_weights((size_weights, score_bias))
    #     if add_noise:
    #         for l in model.layers:
    #             for w in l.weights:
    #                 w.assign_add((np.random.random(size=w.shape) * 0.2) - 0.1)
    # else:
    #     print("Keeping random weights")
    model.summary()
    return model, sizes, prediction_shapes
