import tensorflow as tf


def keras_to_tf_lite(keras_model_path: str, out_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-keras-model',
                        required=True,
                        type=str,
                        dest="input")
    parser.add_argument('-o', '--output-tf-lite-model',
                        required=True,
                        type=str,
                        dest="output")
    args = parser.parse_args()

    keras_to_tf_lite(args.input, args.output)

