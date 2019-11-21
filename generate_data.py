import cv2
import os
import numpy as np
import multiprocessing

from load_yolo_data import RGB_AVERAGE, RGB_STD

RGB_YELLOW_BASE = np.array([216, 197, 41])
RGB_YELLOW_STD = np.array([10, 10, 3])


def generate_image(w, h, sign_sizes, nb_diamond=1, obj_background=500, other_yellow=3):
    im = np.zeros((w, h, 3), dtype=np.uint8)
    im += RGB_AVERAGE.astype(np.uint8)

    for i in range(obj_background):
        c = np.random.randint(0, h), np.random.randint(0, w)
        p = np.random.randint(3, 10)
        min_r, max_r = sorted(np.abs(np.random.normal(20, 30, 2)) * ((obj_background * 1.2 - i) / obj_background))
        poly = get_random_poly(max_r, min_r, p, c)
        color = (RGB_AVERAGE + RGB_STD * np.random.normal(0.0, 1.3, 3)).astype(np.uint8)
        color = tuple([int(c) for c in color])
        im = cv2.fillConvexPoly(im, poly, color)

    diamond_obj = []

    for i in range(nb_diamond):
        while True:
            s = np.random.choice(sign_sizes, 1)
            r = s // 2
            c = np.array((np.random.randint(r, h - r), np.random.randint(r, w - r)))
            good = True
            for pos, size in diamond_obj:
                d = np.linalg.norm(pos - c, 1)
                if d < (s + size) / 2 + 2:
                    good = False
                    break
            if good:
                break
        diamond_obj.append((c, s))
        poly = get_diamond_angles(s, c)
        color = (RGB_YELLOW_BASE + RGB_YELLOW_STD * np.random.normal(0.0, 1.0, 3)).astype(np.uint8)
        color = tuple([int(c) for c in color])
        im = cv2.fillConvexPoly(im, poly, color)

    for i in range(other_yellow):
        while True:
            s = np.random.choice(sign_sizes, 1)
            r = s // 2
            c = np.array((np.random.randint(r, h - r), np.random.randint(r, w - r)))
            good = True
            for pos, size in diamond_obj:
                d = np.linalg.norm(pos - c, 2)
                if d < (s + size) / 2:
                    good = False
                    break
            if good:
                break
        color = (RGB_YELLOW_BASE + RGB_YELLOW_STD * np.random.normal(0.0, 1.0, 3)).astype(np.uint8)
        color = tuple([int(c) for c in color])
        im = cv2.rectangle(im, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), color, -1)

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # cv2.imshow("example", im)
    # cv2.waitKey()
    return im, diamond_obj


def get_diamond_angles(size, center):
    r = int(size / 2)
    top = center[0], center[1] - r
    bot = center[0], center[1] + r
    left = center[0] - r, center[1]
    right = center[0] + r, center[1]
    return np.array([top, right, bot, left], dtype=np.int)


def get_random_poly(max_r, min_r, point_count, center):
    angles = sorted(np.random.uniform(low=0, high=np.pi*2, size=point_count))
    radius = np.random.uniform(low=min_r, high=max_r, size=point_count)

    return np.array([pol2cart(a, r, center[0], center[1]) for a, r in zip(angles, radius)], dtype=np.int)


def pol2cart(theta, rho, cx=0, cy=0):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x + cx, y + cy


def save_yolo_annotations(file_name, diamond_objs, image_size):
    with open(file_name, 'w') as f:
        for center, size in diamond_objs:
            x = float(center[0] / image_size[1])
            y = float(center[1] / image_size[0])
            w = float(size / image_size[0])
            h = float(size / image_size[1])
            line = "0 {} {} {} {}\n".format(x, y, h, w)
            f.write(line)


def generate_data(j):
    im, a = generate_image(220, 400, [6, 10, 14, 24, 42, 84], nb_diamond=5)
    cv2.imwrite(os.path.join(base_dir, "{:07d}.jpg".format(j)), im)
    save_yolo_annotations(os.path.join(base_dir, "{:07d}.txt".format(j)), a, [220, 400])


if __name__ == '__main__':
    count = 20000
    base_dir = '/tmp/data_{}_2'.format(count)
    os.makedirs(base_dir, exist_ok=True)
    pool = multiprocessing.Pool()

    pool.map(generate_data, range(count), chunksize=count//12)



