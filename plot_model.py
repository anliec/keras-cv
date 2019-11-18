import tensorflow as tf

from load_network import load_network


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

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


if __name__ == '__main__':
    plot_model()




