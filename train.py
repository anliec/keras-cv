import numpy as np
import cv2
import os
import shutil
import json
import random
from datetime import datetime
import multiprocessing
from time import time
from load_yolo_data import list_data_from_dir, YoloDataLoader, read_yolo_image, RGB_AVERAGE, RGB_STD
from load_network import load_network
from loss import SSDLikeLoss
from detection_processing import process_detection, draw_roi, Roi, DetectionProcessor
from callback import MAP_eval

from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

sns.set()


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


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def generate_grid_images(shapes: list, sizes: list, class_count: int, input_shape, out_dir: str):
    detection_processor = DetectionProcessor(sizes=sizes, shapes=shapes, image_size=input_shape, threshold=0.5,
                                             nms_threshold=1.1)

    raws = [np.zeros(shape=list(s.astype(np.int)) + [class_count], dtype=np.float16)
            for s in shapes]
    # set the background class to one on every predictions
    for r in raws:
        r[:, :, 0] = 1.0

    for i, r in enumerate(raws):
        r[:, :, 1] = 1.0
        r[:, :, 0] = 0.0
        concat_flatten_raws = np.concatenate([a.reshape(-1, class_count) for a in raws], axis=0)

        pred_roi = detection_processor.process_detection(concat_flatten_raws.reshape((1,) + concat_flatten_raws.shape),
                                                         pool=None)
        bb_im = np.zeros(input_shape + (3,), dtype=np.uint8) + 57
        for roi in pred_roi[0]:
            bb_im = draw_roi(bb_im, [roi], width=1, color=random_color())
        bb_im = cv2.cvtColor(bb_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, "boxes_layers_{}.png".format(i)), bb_im)
        r[:, :, 1] = 0.0
        r[:, :, 0] = 1.0


def train(data_path: str, batch_size: int = 2, epoch: int = 1, learning_rate=0.01, base_weight=None):
    # setup tensorflow backend (prevent "Blas SGEMM launch failed" error)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    config = {"size_value": [110, 200], "dropout_rate": 0.0, "dropout_strategy": "all",
              "layers_filters": (16, 16, 24, 24), "expansions": (1, 6, 6), 'use_mobile_net': False,
              "use_resnet": True}

    # save config
    with open("config.json", 'w') as c:
        json.dump(config, c)

    # [451, 579]
    model, sizes, shapes = load_network(**config)
    # plot_model(model, to_file="model.png", show_shapes=False, show_layer_names=True)
    input_shape = model.input.shape[1:3]
    input_shape = int(input_shape[0]), int(input_shape[1])

    if base_weight is not None:
        model.load_weights(base_weight)

    # generate_grid_images(shapes, sizes, class_count=2, input_shape=input_shape, out_dir=".")

    if os.path.isfile("data.json"):
        with open("data.json", 'r') as j:
            data = json.load(j)
        images_list_train = data["train"]
        images_list_test = data["val"]
    else:
        images_list = list_data_from_dir(data_path, "*.jpg")
        if os.path.isdir("data/test"):
            test_images_list = list_data_from_dir("data/test", "*.jpg")
        else:
            test_images_list = []

        split = int(round(len(images_list) * 0.9))
        images_list_train = images_list[:split]
        images_list_test = images_list[split:] + test_images_list

        with open("data.json", 'w') as j:
            json.dump({"train": images_list_train, "val": images_list_test}, j)

    loss = SSDLikeLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                  momentum=0.8,
                                  decay=1e-2/epoch)
    model.compile(optimizer=opt,
                  loss=loss.compute_loss
                  )

    pool = multiprocessing.Pool()

    train_sequence = YoloDataLoader(images_list_train, batch_size, input_shape, shapes,
                                    pyramid_size_list=sizes, disable_augmentation=False,
                                    movement_range_width=0.05, movement_range_height=0.05,
                                    zoom_range=(0.7, 1.1), flip=True, brightness_range=(0.7, 1.3),
                                    use_multiprocessing=True, pool=pool)
    # train_sequence.preload_data()
    test_sequence = YoloDataLoader(images_list_test, batch_size, input_shape, shapes,
                                   pyramid_size_list=sizes, disable_augmentation=True)
    # test_sequence.preload_data()

    map_callback = MAP_eval(test_sequence, sizes, shapes, input_shape, detection_threshold=0.5, mns_threshold=0.3,
                            iou_thresholds=(0.25, 0.5, 0.75), frequency=10, epoch_start=min(epoch//2, 25))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)
    log_dir = "logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)

    history = model.fit_generator(train_sequence, validation_data=test_sequence, epochs=epoch, shuffle=True,
                                  use_multiprocessing=False,
                                  callbacks=[map_callback, tensorboard_callback, early_stopping])

    model.save("model.h5")

    plot_history(history, "nNet")

    detection_processor = DetectionProcessor(sizes=sizes, shapes=shapes, image_size=input_shape, threshold=0.5,
                                             nms_threshold=0.3)
    out_dir = "debug/"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    durations = []
    prediction_count = 0
    # gt_count = 0
    fps = 1
    fps_nn = 1
    for i, (x_im, y_raw) in enumerate(test_sequence.data_list_iterator()):
        seconds_left = (len(test_sequence.image_list) - i) / fps
        print("Processing Validation Frame {:4d}/{:d}  -  {:.2f} fps  ETA: {} min {} sec (NN: {:.2f} fps)"
              "".format(i, len(test_sequence.image_list), fps, int(seconds_left // 60), int(seconds_left) % 60, fps_nn),
              end="\r")
        f_start = time()
        x = x_im.reshape((1,) + x_im.shape)
        # predict result for the image
        start = time()
        raw_pred = model.predict(x)
        end = time()

        fps_nn = 1 / (end - start)
        durations.append(end - start)
        # process detection
        pred_roi = detection_processor.process_detection(raw_pred, pool=None)
        # draw detections
        bb_im = ((x_im * RGB_STD) + RGB_AVERAGE).astype(np.uint8)
        bb_im = draw_roi(bb_im, pred_roi[0][:100])
        bb_im = cv2.cvtColor(bb_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, "{:03d}_im.jpg".format(i)), bb_im)
        prediction_count += len(pred_roi[0])
        # process gt
        # pred_roi = detection_processor.process_detection(y_raw.reshape((1,) + y_raw.shape), pool=None)
        # # draw gt
        # bb_im = ((x_im * RGB_STD) + RGB_AVERAGE).astype(np.uint8)
        # bb_im = draw_roi(bb_im, pred_roi[0])
        # bb_im = cv2.cvtColor(bb_im, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(out_dir, "gt_{:03d}_im.jpg".format(i)), bb_im)
        # gt_count += len(pred_roi[0])
        f_end = time()
        fps = 1 / (f_end - f_start)

    print()
    print("Prediction done in {}s ({} fps)".format(sum(durations), len(images_list_test) / sum(durations)))
    print("Fastest: {:.4f}s  {:.2f} fps".format(min(durations), 1/min(durations)))
    print("Slowest: {:.4f}s  {:.2f} fps".format(max(durations), 1/max(durations)))

    with open("results.json", 'w') as f:
        json.dump({"config": config,
                   "nn_fps": 1 / np.mean(durations),
                   "prediction_count": prediction_count,
                   "last_mAP": map_callback.maps[-1],
                   "mAPs": list(zip(map_callback.epochs, map_callback.maps)),
                   "stats": list(zip(map_callback.epochs, map_callback.scores))
                   },
                  f)

    # Save a visualisation of the first layer
    # first_conv_weights = model.layers[2].get_weights()[0]
    # for i in range(first_conv_weights.shape[3]):
    #     f = first_conv_weights[:, :, :, i]
    #     for l in range(3):
    #         f[:, :, l] -= f[:, :, l].min()
    #         f[:, :, l] *= 255 / f[:, :, l].max()
    #     f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("Conv1_filter{}.png".format(i), f)


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
    parser.add_argument('-lr', '--learning-rate',
                        required=False,
                        type=float,
                        default=0.01,
                        dest="lr")
    parser.add_argument('-m', '--base-weights',
                        required=False,
                        type=str,
                        default=None,
                        dest="weights")
    args = parser.parse_args()

    train(args.data_path, args.batch, args.epoch, args.lr, args.weights)

