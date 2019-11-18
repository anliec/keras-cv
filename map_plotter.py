import glob
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns

sns.set()


def plot_map(resutls_dir: str, output_graph_file: str):
    speed_files = glob.glob(os.path.join(resutls_dir, "**", "avg_lite_speed"), recursive=True)
    speeds = []
    maps = defaultdict(list)
    for s in speed_files:
        d = os.path.dirname(s)
        results_file = os.path.join(d, "results.json")
        if not os.path.isfile(results_file):
            print("Unable to find {}, skipping".format(results_file))
            continue
        with open(s, 'r') as f:
            avg_inference_time = float(f.readline()) * 10e-6
        try:
            with open(results_file, 'r') as f:
                stats = json.load(f)
        except json.decoder.JSONDecodeError:
            print("Unable to read json {}".format(results_file))
            continue
        speeds.append(avg_inference_time)
        th_best_map = defaultdict(int)
        for e, m in stats['mAPs']:
            for th, v in m.items():
                th_best_map[th] = max(th_best_map[th], v)
        for th, v in th_best_map.items():
            maps[th].append(v)

    speed_sorted = np.array(sorted(zip(speeds, *maps.values()), key=lambda t: t[0]))

    for i, k in enumerate(maps.keys()):
        plt.plot(speed_sorted[:, 0], speed_sorted[:, i + 1], "+", label="mAP@{}".format(k))

    plt.title("Evolution of mAP with speed for different networks structure")
    plt.xlabel("Inference speed (s)")
    plt.ylabel("mAP (%)")
    plt.legend()
    plt.savefig(output_graph_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('grid_search_dir',
                        type=str)
    parser.add_argument('figure',
                        type=str)
    args = parser.parse_args()

    plot_map(args.grid_search_dir, args.figure)

