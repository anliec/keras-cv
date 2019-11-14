import glob
import os
import json
import numpy as np
from matplotlib import pyplot as plt


def plot_map(resutls_dir: str, output_graph_file: str):
    speed_files = glob.glob(os.path.join(resutls_dir, "**", "avg_lite_speed"), recursive=True)
    speeds = []
    maps = []
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
        nn_maps = np.array(stats['mAPs'])
        maps.append(nn_maps.max(axis=0)[1])

    plt.plot(speeds, maps, 'b+')
    plt.xlabel("Inference speed (s)")
    plt.ylabel("mAP (%)")
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

