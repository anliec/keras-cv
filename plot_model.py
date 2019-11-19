import tensorflow as tf

from load_network import load_network


def get_last_conv_ancestor(model: tf.keras.Model, layer: tf.keras.layers.Layer):
    prev_name = layer.input.name.split('/')[0]
    try:
        layer = model.get_layer(name=prev_name)
    except ValueError:
        return None
    if type(layer) == tf.keras.layers.Conv2D:
        return layer
    else:
        return get_last_conv_ancestor(model, layer)


def compute_flops(input_shape, conv_filter, stride, padding=1, activation='relu') -> int:
    """
    from https://stats.stackexchange.com/questions/291843/how-to-understand-calculate-flops-of-the-neural-network-model
    :return: flops of the given layer
    """
    # input_shape = (3, 300, 300)  # Format:(channels, rows,cols)
    # conv_filter = (64, 3, 3, 3)  # Format: (num_filters, channels, rows, cols)
    # stride = 1
    # padding = 1
    # activation = 'relu'
    conv_filter = conv_filter[::-1]
    input_shape = input_shape[::-1]

    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    flops_per_instance = n + (n - 1)  # general definition for number of flops (n: multiplications and n-1: additions)

    num_instances_per_filter = ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
    num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters

    if activation.lower() == 'relu' or activation.lower() == 'relu6':
        # Here one can add number of flops required
        # Relu takes 1 comparison and 1 multiplication
        # Assuming for Relu: number of flops equal to length of input vector
        total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]
    return total_flops_per_layer


def plot_model():
    # setup tensorflow backend (prevent "Blas SGEMM launch failed" error)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    config = {"size_value": [110, 200], "dropout_rate": 0.2, "dropout_strategy": "last",
              "layers_filters": (16, 16, 24, 24), "expansions": (1, 6, 6)}

    model, sizes, shapes = load_network(**config)

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    bn = ""
    activation = "Linear"

    lines = []
    flops = 0

    for l in model.layers[::-1]:
        if type(l) == tf.keras.layers.BatchNormalization:
            bn = "\\checkmark"
        elif type(l) == tf.keras.layers.ReLU:
            activation = "ReLu6"
        elif type(l) == tf.keras.layers.Softmax:
            activation = "Softmax"
        elif type(l) == tf.keras.layers.Conv2D:
            filters = l.filters
            kernel_size = l.kernel_size
            strides = l.strides
            flops += compute_flops(l.input.shape[1:], l.kernel.shape, strides[0], padding=1, activation=activation)
            name = l.name.replace('_', '\\_')
            ancestor = get_last_conv_ancestor(model, l)
            if ancestor is None:
                in_name = ""
            else:
                in_name = ancestor.name.replace('_', '\\_')
            name = name.replace('\\_conv', '')
            in_name = in_name.replace('\\_conv', '')
            lines.append("{} & ${}$ & ${}$ & ${}$ & {} & {} & ${}$ & {} \\\\"
                         "".format(name, filters, kernel_size,
                                   strides, activation, bn, l.input.shape[1:], in_name))
            bn = ""
            activation = "Linear"

    print("\n".join(lines[::-1]))
    print("Total Flops = {}".format(flops))


if __name__ == '__main__':
    plot_model()




