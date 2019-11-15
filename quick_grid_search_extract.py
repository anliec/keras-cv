import json
import glob
import numpy as np
import os
from collections import defaultdict


def extract_from_grid_search(results_dir: str):
    results = glob.glob(os.path.join(results_dir, "**", "results.json"))
    for r in sorted(results):
        try:
            with open(r, 'r') as j:
                stats = json.load(j)
        except Exception:
            print("Unable to open file {}".format(r))
            continue
        name = os.path.basename(os.path.dirname(r))
        maps = defaultdict(int)
        for e, m in stats['mAPs']:
            for th, v in m.items():
                maps[th] = max(maps[th], v)
        print(name)
        for th, v in maps.items():
            print(" mAP@{:.2f}: {:.2f}".format(float(th), v))
        print(stats['config'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('results_path',
                        type=str)
    args = parser.parse_args()

    extract_from_grid_search(args.results_path)
