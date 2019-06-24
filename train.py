import numpy as np
import cv2
import os

from load_yolo_data import list_data_from_dir, YoloDataLoader, data_list_iterator
from load_network import load_network
from loss import detection_loss
from detection_processing import process_detection, draw_roi, Roi


def train(data_path: str):

    model = load_network(input_shape=[902, 1158])
    input_shape = model.layers[0].input_shape[1:3]
    annotation_shape = model.layers[-1].output_shape[1:3]
    annotation_shape = int(annotation_shape[0]), int(annotation_shape[1])

    images_list = list_data_from_dir(data_path, "*.jpg")

    split = int(round(len(images_list) * 0.9))

    images_list_train = images_list[:split]
    images_list_test = images_list[split:]

    model.compile(optimizer='sgd',
                  loss=detection_loss(score_min_bound=0.1, gaussian_diameter=11),
                  # loss_weights={'Score': 1.0, 'Size': 0.0000001})
                  loss_weights={'Score': 1.0, 'Size': 0.0})

    model.fit_generator(YoloDataLoader(images_list_train, 4, input_shape, annotation_shape),
                        validation_data=YoloDataLoader(images_list_test, 4, input_shape, annotation_shape),
                        epochs=1, shuffle=True)

    out_dir = "debug/"
    os.makedirs(out_dir, exist_ok=True)
    for i, (x_im, score, size) in enumerate(data_list_iterator(images_list_test, input_shape, annotation_shape)):
        score_pred, size_pred = model.predict(x_im.reshape((1,) + x_im.shape))
        pred_norm = score_pred - score_pred.min()
        pred_norm = pred_norm / pred_norm.max()
        pred_norm = (pred_norm[0, :, :, 0] * 255).astype(np.uint8)
        pred_norm = cv2.resize(pred_norm, (x_im.shape[1], x_im.shape[0]))
        cv2.imwrite(os.path.join(out_dir, "{:03d}_pred.png".format(i)), pred_norm)
        pred_roi = process_detection(score_pred, size_pred, 0.95 * score_pred.max())
        bb_im = (x_im * 255 / x_im.max()).astype(np.uint8)
        bb_im = draw_roi(bb_im, pred_roi)
        bb_im = cv2.cvtColor(bb_im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_dir, "{:03d}_im.jpg".format(i)), bb_im)
        del score_pred, size_pred, pred_norm


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='Path to the input training data')
    args = parser.parse_args()

    train(args.data_path)

