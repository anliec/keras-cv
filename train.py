import numpy as np
import cv2
import os

from load_yolo_data import list_data_from_dir, YoloDataLoader, RawYoloDataLoader
from load_network import load_network
from loss import detection_loss, raw_output_loss
from detection_processing import process_detection, draw_roi, Roi, process_detection_raw
from metrics import map_metric


def train(data_path: str, batch_size: int = 2, epoch: int = 1, random_init: bool = False):
    model, sizes = load_network(size_value=[451, 579], random_init=random_init, pyramid_depth=6,
                                first_pyramid_output=1)
    input_shape = model.layers[0].input_shape[1:3]
    annotation_shape = model.layers[-1].output_shape[1:3]
    annotation_shape = int(annotation_shape[0]), int(annotation_shape[1])

    images_list = list_data_from_dir(data_path, "*.jpg")

    split = int(round(len(images_list) * 0.9))

    images_list_train = images_list[:split]
    images_list_test = images_list[split:]

    # model.compile(optimizer='sgd',
    #               loss=detection_loss(score_min_bound=0.1, gaussian_diameter=5),
    #               loss_weights={'Score': 1.0, 'Size': 0.000001}
    #               # loss_weights={'Score': 1.0, 'Size': 0.0}
    #               )
    # model.fit_generator(YoloDataLoader(images_list_train, batch_size, input_shape, annotation_shape),
    #                     validation_data=YoloDataLoader(images_list_test, batch_size, input_shape, annotation_shape),
    #                     epochs=epoch, shuffle=True)
    model.compile(optimizer='sgd',
                  loss=raw_output_loss(score_min_bound=0.1, gaussian_diameter=11, gaussian_height=3),
                  # metrics=[map_metric()]
                  )

    train_sequence = RawYoloDataLoader(images_list_train, batch_size, input_shape, annotation_shape,
                                       pyramid_size_list=sizes)
    test_sequence = RawYoloDataLoader(images_list_test, batch_size, input_shape, annotation_shape,
                                      pyramid_size_list=sizes)

    model.fit_generator(train_sequence, validation_data=test_sequence, epochs=epoch, shuffle=True)

    out_dir = "debug/"
    os.makedirs(out_dir, exist_ok=True)
    for i, (x_im, raw) in enumerate(test_sequence.data_list_iterator()):
        raw_pred = model.predict(x_im.reshape((1,) + x_im.shape))
        pred_norm = raw_pred - raw_pred.min()
        pred_norm = pred_norm / pred_norm.max()
        pred_norm_max = np.max(pred_norm, axis=3)
        pred_norm_max = (pred_norm_max[0, :, :, 0] * 255).astype(np.uint8)
        pred_norm_max = cv2.resize(pred_norm_max, (x_im.shape[1], x_im.shape[0]))
        cv2.imwrite(os.path.join(out_dir, "{:03d}_pred_max.png".format(i)), pred_norm_max)
        pred_roi = process_detection_raw(raw_pred, sizes, max(0.001, 0.95 * raw_pred.max()))
        bb_im = (x_im * 255 / x_im.max()).astype(np.uint8)
        bb_im = draw_roi(bb_im, pred_roi)
        bb_im = cv2.cvtColor(bb_im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_dir, "{:03d}_im.jpg".format(i)), bb_im)


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
    parser.add_argument('-r', '--random-weights',
                        required=False,
                        help='Initialise weights with random values',
                        dest="random_init",
                        action='store_true')
    args = parser.parse_args()

    train(args.data_path, args.batch, args.epoch, args.random_init)

