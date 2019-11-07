import numpy as np
import cv2
import os
import json
import shutil
import random
import multiprocessing
from time import time
from load_yolo_data import list_data_from_dir, YoloDataLoader, read_yolo_image, RGB_AVERAGE, RGB_STD
from load_network import load_network
from loss import SSDLikeLoss
from detection_processing import process_detection, draw_roi, Roi, DetectionProcessor
from metrics import map_metric

from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')


TO_EXPLORE = {
    "dropout_rate": [0.1, 0.5, 0.8],
    "dropout_strategy": ["all", "last"],
    "layers_filters": [(32, 16, 24, 32), (16, 16, 24, 32), (16, 16, 24, 24), (16, 16, 16, 24), (16, 8, 16, 24),
                       (8, 8, 16, 24), (8, 8, 16, 16), (8, 8, 8, 16), (8, 8, 8, 8)],
    "expansions": [(1, 6, 6), (1, 1, 1), (1, 3, 3)]
}


def generate_combinations():
    count = 1
    for k, v in TO_EXPLORE.items():
        count *= len(v)
    print("{} combinations to explore".format(count))
    cur_pos = {k: 0 for k in TO_EXPLORE.keys()}
    while True:
        kwargs = {k: TO_EXPLORE[k][pos] for k, pos in cur_pos.items()}
        if kwargs["dropout_strategy"] != "all" or kwargs["dropout_rate"] <= 0.2:
            yield kwargs
        for k, v in cur_pos.items():
            if v + 1 < len(TO_EXPLORE[k]):
                cur_pos[k] += 1
                break
            else:
                cur_pos[k] = 0
        else:
            break


# from https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def plot_history(history, base_name=""):
    plt.clf()
    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.savefig(base_name + "accuracy.png")
    # plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    axes = plt.gca()
    axes.set_xlim([0, None])
    axes.set_ylim([0, None])
    plt.savefig(base_name + "loss.png")
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    axes = plt.gca()
    min_loss = min(min(history.history['loss']), min(history.history['val_loss']))
    loss_lim = min_loss * 2
    axes.set_ylim([0, loss_lim])
    axes.set_xlim([0, None])
    plt.savefig(base_name + "loss_zoomed.png")
    plt.clf()


def grid_search(data_path: str, batch_size: int = 2, epoch: int = 1):
    model, sizes, shapes = load_network(size_value=[110, 200])

    input_shape = model.input.shape[1:3]
    input_shape = int(input_shape[0]), int(input_shape[1])

    images_list = list_data_from_dir(data_path, "*.jpg")
    if os.path.isdir("data/test"):
        test_images_list = list_data_from_dir("data/test", "*.jpg")
    else:
        test_images_list = []

    split = int(round(len(images_list) * 0.9))
    images_list_train = images_list[:split]
    images_list_test = images_list[split:] + test_images_list

    images_list_train = images_list_train[128:]
    images_list_test = images_list_test[100:]

    loss = SSDLikeLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    pool = multiprocessing.Pool()

    train_sequence = YoloDataLoader(images_list_train, batch_size, input_shape, shapes,
                                    pyramid_size_list=sizes, disable_augmentation=False,
                                    movement_range_width=0.2, movement_range_height=0.2,
                                    zoom_range=(0.7, 1.1), flip=True, brightness_range=(0.5, 1.5),
                                    use_multiprocessing=True, pool=pool)
    test_sequence = YoloDataLoader(images_list_test, batch_size, input_shape, shapes,
                                   pyramid_size_list=sizes, disable_augmentation=True)

    detection_processor = DetectionProcessor(sizes=sizes, shapes=shapes, image_size=input_shape, threshold=0.5,
                                             nms_threshold=0.3)

    for comb, kwargs in enumerate(generate_combinations()):
        cur_dir = os.path.join("grid_search", "test_{}".format(comb))
        os.makedirs(cur_dir, exist_ok=False)

        with open(os.path.join(cur_dir, "config.json"), 'w') as f:
            json.dump(kwargs, f, indent=4)

        model, sizes, shapes = load_network(size_value=[110, 200], **kwargs)

        model.compile(optimizer='sgd',
                      loss=loss.compute_loss
                      )

        history = model.fit_generator(train_sequence, validation_data=test_sequence, epochs=epoch, shuffle=True,
                                      use_multiprocessing=False)

        plot_history(history, os.path.join(cur_dir, "nNet"))

        arrays = {k: np.array(v) for k, v in history.history.items()}
        np.savez(os.path.join(cur_dir, "history.npz"), arrays)
        model.save(os.path.join(cur_dir, "model.h5"))

        # Save a visualisation of the first layer
        os.makedirs(os.path.join(cur_dir, "filters"), exist_ok=False)
        first_conv_weights = model.layers[2].get_weights()[0]
        for i in range(first_conv_weights.shape[3]):
            f = first_conv_weights[:, :, :, i]
            for l in range(3):
                f[:, :, l] -= f[:, :, l].min()
                f[:, :, l] *= 255 / f[:, :, l].max()
            f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cur_dir, "filters", "Conv1_filter{}.png".format(i)), f)

        durations = []
        prediction_count = 0
        fps = 1
        fps_nn = 1
        fps_nn_list = []
        os.makedirs(os.path.join(cur_dir, "frames"), exist_ok=False)
        for i, (x_im, y_raw) in enumerate(test_sequence.data_list_iterator()):
            seconds_left = (len(test_sequence.image_list) - i) / fps
            print("Processing Validation Frame {:4d}/{:d}  -  {:.2f} fps  ETA: {} min {} sec (NN: {:.2f} fps)"
                  "".format(i, len(test_sequence.image_list), fps, int(seconds_left // 60), int(seconds_left) % 60,
                            fps_nn),
                  end="\r")
            f_start = time()
            x = x_im.reshape((1,) + x_im.shape)
            # predict result for the image
            start = time()
            raw_pred = model.predict(x)
            end = time()

            fps_nn = 1 / (end - start)
            fps_nn_list.append(fps_nn)
            durations.append(end - start)
            # process detection
            pred_roi = detection_processor.process_detection(raw_pred, pool=None)
            # draw detections
            bb_im = ((x_im * RGB_STD) + RGB_AVERAGE).astype(np.uint8)
            bb_im = draw_roi(bb_im, pred_roi[0][:100])
            bb_im = cv2.cvtColor(bb_im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cur_dir, "frames", "{:03d}_im.jpg".format(i)), bb_im)
            prediction_count += len(pred_roi[0])
            f_end = time()
            fps = 1 / (f_end - f_start)

        with open(os.path.join(cur_dir, "results.json"), 'w') as f:
            json.dump({"config": kwargs,
                       "nn_fps": fps_nn_list,
                       "prediction_count": prediction_count,
                       "flops": get_flops(model)},
                      f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='Path to the input training data')
    parser.add_argument('-b', '--batch-size',
                        required=False,
                        type=int,
                        default=2,
                        help='Size a of batch of data send to the neural network at one time',
                        dest="batch")
    parser.add_argument('-e', '--number-of-epoch',
                        required=False,
                        type=int,
                        default=1,
                        help='Number of epoch during training',
                        dest="epoch")
    args = parser.parse_args()

    grid_search(args.data_path, args.batch, args.epoch)















