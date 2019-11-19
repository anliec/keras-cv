import sys
import math

sys.path.append('/home/nicolas/Programation/keras-cv/PlotNeuralNet')
from PlotNeuralNet.pycore.tikzeng import *

from load_network import load_network
import tensorflow as tf


def plot_model():
    # setup tensorflow backend (prevent "Blas SGEMM launch failed" error)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    config = {"size_value": [110, 200], "dropout_rate": 0.1, "dropout_strategy": "all",
              "layers_filters": (16, 16, 24, 24), "expansions": (1, 6, 6)}

    model, sizes, shapes = load_network(**config)
    forward_offset = 7
    output_offsets = [
        (forward_offset, 10, -6),
        (forward_offset, 10, 6),
        (forward_offset, 10, -8),
        (forward_offset, 10, 0),
        (forward_offset, 10, 8)
    ]
    size_factor = 3

    output_offsets = ["({})".format(",".join([str(v / size_factor) for v in t])) for t in output_offsets]
    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
        to_input("/home/nicolas/Programation/keras-cv/data/test/test.jpg", width=40 / size_factor,
                 height=22 / size_factor, name="input")
    ]

    for layer in model.layers:
        if type(layer) == tf.keras.layers.Conv2D:
            ancestor = get_last_conv_ancestor(model, layer)
            if ancestor is None:
                ancestor_name = "input"
            else:
                ancestor_name = ancestor.name
            if layer.name.split('_')[0] == "output":
                output_id = int(layer.name.split('_')[1]) - 1
                arch.append(to_Conv(layer.name,
                                    n_filer=layer.filters,
                                    s_filer=layer.input.shape[2],
                                    width=math.log(layer.filters, 2) * 3 / size_factor,
                                    height=layer.output.shape[1] / size_factor,
                                    depth=layer.output.shape[2] / size_factor,
                                    to="({}-east)".format(ancestor_name),
                                    offset=output_offsets[output_id]))
                arch.append(to_connection(ancestor_name, layer.name))
            else:
                if layer.strides != (1, 1):
                    pool_name = layer.name + "-stride"
                    arch.append(to_Pool(pool_name, offset="({},0,0)".format(forward_offset / size_factor),
                                        height=layer.input.shape[1] / size_factor,
                                        depth=layer.input.shape[2] / size_factor,
                                        to="({})".format(ancestor_name if ancestor_name == "input"
                                                         else ancestor_name + "-east")))
                    if ancestor_name != "input":
                        arch.append(to_connection(ancestor_name, pool_name))
                    ancestor_name = pool_name
                arch.append(to_Conv(layer.name,
                                    n_filer=layer.filters,
                                    s_filer=layer.output.shape[2],
                                    width=math.log(layer.filters, 2) * 3 / size_factor,
                                    height=layer.output.shape[1] / size_factor,
                                    depth=layer.output.shape[2] / size_factor,
                                    to="({})".format(ancestor_name if ancestor_name == "input"
                                                     else ancestor_name + "-east"),
                                    offset="({},0,0)".format(0)))

    arch += [to_end()]

    return arch


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


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    arch = plot_model()
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
