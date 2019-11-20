import tensorflow as tf
import json
import os

from load_network import load_network
from load_yolo_data import list_data_from_dir, read_yolo_image
import random


def keras_to_tf_lite(keras_model_path: str, out_path: str, data_path: str, data_limit: int = None,
                     config_file: str = None) -> None:
    if config_file is None and os.path.isfile(os.path.join(os.path.dirname(keras_model_path), "config.json")):
        config_file = os.path.join(os.path.dirname(keras_model_path), "config.json")
    if config_file is not None:
        with open(config_file, 'r') as c:
            config = json.load(c)
    else:
        config = {}

    model, sizes, shapes = load_network(**config)

    if os.path.isfile(keras_model_path):
        model.load_weights(keras_model_path)
    else:
        print("Warning: no weights were loaded, random weights were used")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # quantification data provided run full quantification
    if data_path is not None:
        input_shape = model.input.shape[1:3]
        input_shape = int(input_shape[0]), int(input_shape[1])
        images_list = list_data_from_dir(data_path, "*.jpg")
        if data_limit is not None:
            random.shuffle(images_list)
            images_list = images_list[:data_limit]

        def representative_dataset_gen():
            for image_path in images_list:
                im = read_yolo_image(image_path, input_shape)
                im = im.reshape((1,) + im.shape)
                yield [im]

        converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-keras-model',
                        required=True,
                        type=str,
                        dest="model")
    parser.add_argument('-c', '--input-model-config',
                        required=False,
                        type=str,
                        default=None,
                        dest="config")
    parser.add_argument('-o', '--output-tf-lite-model',
                        required=True,
                        type=str,
                        dest="output")
    parser.add_argument('-d', '--quantification-data',
                        required=False,
                        default=None,
                        type=str,
                        dest="quantification_data")
    parser.add_argument('-l', '--data-limit',
                        required=False,
                        type=int,
                        default=None,
                        dest="limit")
    args = parser.parse_args()

    keras_to_tf_lite(args.model, args.output, args.quantification_data, args.limit, args.config)

