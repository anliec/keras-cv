import tensorflow as tf
import json

from load_yolo_data import YoloDataLoader
from detection_processing import DetectionProcessor
from load_network import load_network


def eval_map(validation_data: YoloDataLoader, model: tf.keras.Model, processor: DetectionProcessor,
             iou_thresholds=(0.5,)):
    fn_roi = []
    matches = []
    min_iou_th = min(iou_thresholds)

    for i in range(len(validation_data)):
        x_val, y_true = validation_data[i]
        y_pred = model.predict(x_val)
        y_pred_roi = processor.process_detection(y_pred)
        y_true_roi = processor.process_detection(y_true)
        for im_y_pred_roi, im_y_true_roi in zip(y_pred_roi, y_true_roi):
            for pred_roi in sorted(im_y_pred_roi, key=lambda roi: -roi.confidence):
                best = None
                best_iou = min_iou_th
                best_i = None
                for true_roi_index, true_roi in enumerate(im_y_true_roi):
                    iou = pred_roi.get_overlap(true_roi)
                    if iou > best_iou:
                        best = true_roi
                        best_iou = iou
                        best_i = true_roi_index
                if best is not None:
                    matches.append((pred_roi, best, best_iou))
                    del im_y_true_roi[best_i]
                else:
                    matches.append((pred_roi, None, None))
            fn_roi += im_y_true_roi

    gt_count = len([tp for tp in matches if tp[1] is not None]) + len(fn_roi)
    rolling_tp_counts = {k: [] for k in iou_thresholds}
    rolling_fp_counts = {k: [] for k in iou_thresholds}
    rolling_precision = {k: [] for k in iou_thresholds}
    rolling_recall = {k: [] for k in iou_thresholds}
    last_tp, last_fp = {k: 0 for k in iou_thresholds}, {k: 0 for k in iou_thresholds}
    for pred_roi, true_roi, iou in sorted(matches, key=lambda m: -m[0].confidence):
        for iou_th in iou_thresholds:
            if true_roi is not None and iou >= iou_th:
                last_tp[iou_th] += 1
            else:
                last_fp[iou_th] += 1

            rolling_tp_counts[iou_th].append(last_tp[iou_th])
            rolling_fp_counts[iou_th].append(last_fp[iou_th])

            precision = last_tp[iou_th] / (last_fp[iou_th] + last_tp[iou_th])
            rolling_precision[iou_th].append(precision)
            rolling_recall[iou_th].append(last_tp[iou_th] / gt_count)

    auc = {}
    for iou_th in iou_thresholds:
        top_precision_by_recall = [(0, 1.0)]
        last_top_precision = 0
        for p, r in zip(rolling_precision[iou_th][::-1], rolling_recall[iou_th][::-1]):
            if last_top_precision < p:
                top_precision_by_recall.append((p, r))
                last_top_precision = p

        top_precision_by_recall.append((last_top_precision, 0.0))

        area_under_curve = sum([p_next[1] * (p_next[0] - p_prev[0]) for p_prev, p_next in
                                zip(top_precision_by_recall[:-1], top_precision_by_recall[1:])])
        auc[iou_th] = area_under_curve

    fn_counts = {k: gt_count - v for k, v in last_tp.items()}
    return auc, last_tp, last_fp, fn_counts


def eval_model_map(model_path: str, model_config_path: str, iou_thresholds, output_map_stat_file_path: str,
                   data_path: str = "data.json", batch_size=8):
    # setup tensorflow backend (prevent "Blas SGEMM launch failed" error)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    with open(model_config_path, 'r') as f:
        kwargs = json.load(f)
    model, sizes, shapes = load_network(**kwargs)
    model.load_weights(model_path)
    with open(data_path, 'r') as j:
        data = json.load(j)
    images_list_test = data["val"]
    input_size = tuple(kwargs["size_value"])
    test_sequence = YoloDataLoader(images_list_test, batch_size, input_size, shapes,
                                   pyramid_size_list=sizes, disable_augmentation=True)
    detection_processor = DetectionProcessor(sizes=sizes, shapes=shapes, image_size=input_size, threshold=0.5,
                                             nms_threshold=0.3)

    print(iou_thresholds)
    mean_ap, tp, fp, fn = eval_map(test_sequence, model, detection_processor, iou_thresholds)
    results = {}
    for th in iou_thresholds:
        print("mAP@{}: {}  TP: {}  FP: {}  FN: {}".format(int(th * 100), mean_ap[th], tp[th], fp[th], fn[th]))
        results[th] = {"mAP": mean_ap[th], "TP": tp[th], "FP": fp[th], "FN": fn[th]}

    with open(output_map_stat_file_path, 'w') as r:
        json.dump(results, r)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        dest='data_path',
                        type=str,
                        required=True,
                        help='Path to the input training data')
    parser.add_argument("-m", dest='model_path',
                        required=True,
                        type=str)
    parser.add_argument("-c", dest='config_path',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument("-o", dest='output_path',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-b', '--batch-size',
                        required=False,
                        type=int,
                        default=8,
                        help='Size a of batch of data send to the neural network at one time',
                        dest="batch")
    parser.add_argument('-th', '--iou-thresholds',
                        required=False,
                        type=float,
                        nargs='+',
                        default=[0.5],
                        dest="iou_thresholds")
    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.model_path), "map_eval.json")

    eval_model_map(args.model_path, args.config_path, args.iou_thresholds, args.output_path, args.data_path, args.batch)



