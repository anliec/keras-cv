import numpy as np
import cv2
import glob
import os
from multiprocessing import Pool
import time


def compute_dataset_values(dataset_path: str, image_extension: str = ".jpg"):
    files = glob.glob(os.path.join(dataset_path, "*" + image_extension))

    pool = Pool()

    before = time.time()
    res = pool.map(get_im_avg_var, files, chunksize=10)
    after = time.time()
    print("Individual results get for every images in {}s\n".format(after - before))

    res = np.array(res, dtype=np.float64)
    avg = res[:, 0]
    var = res[:, 1]

    before = time.time()
    m, v, c = merge_avg_var(avg, var)
    after = time.time()
    print("Merge computed in {}s\n\n".format(after - before))

    print("Computed pixels statistical value over the given dataset!")
    print("Average pixel is: (BGR)")
    print(avg.mean(axis=0))
    print("Variance: (BGR)")
    print(v)
    print("standard deviation: (BGR)")
    print(np.sqrt(v))


def get_im_avg_var(im_path: str):
    im = cv2.imread(im_path)
    im = im.reshape((-1, 3))
    return np.mean(im, axis=0), np.var(im, axis=0)


def merge_avg_var(avg: np.ndarray, var: np.ndarray):
    if avg.shape[0] == 1:
        return avg[0, :], var[0, :], 1

    pivot = round(avg.shape[0] / 2)

    avg1, var1, count1 = merge_avg_var(avg[:pivot, :], var[:pivot, :])
    avg2, var2, count2 = merge_avg_var(avg[pivot:, :], var[pivot:, :])

    # from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    m = (avg1 + avg2) / 2.0
    delta = avg1 - avg2
    m_1 = var1 * (count1 - 1)
    m_2 = var2 * (count2 - 1)
    v = m_1 + m_2 + delta ** 2 * count1 * count2 / (count1 + count2)
    v = v / (count1 + count2 - 1)

    return m, v, count1 + count2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='Path to the input data')
    args = parser.parse_args()

    compute_dataset_values(args.data_path)

