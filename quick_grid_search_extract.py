import json
import glob
import numpy as np
import os


def extract_from_grid_search(results_dir: str, label_to_extract):
    results = glob.glob(os.path.join(results_dir, "**", "results.json"))
    for r in sorted(results):
        try:
            with open(r, 'r') as j:
                stats = json.load(j)
        except Exception:
            print("Unable to open file {}".format(r))
            continue
        name = os.path.basename(os.path.dirname(r))
        maps = np.array(stats['mAPs'])
        print("{}:\t{:.2f}\t{}\t{:.2f}"
              "".format(name, stats["last_mAP"], stats['config']['layers_filters'], np.max(maps, axis=0)[1]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('results_path',
                        type=str)
    parser.add_argument('label_to_extract',
                        type=str)
    args = parser.parse_args()

    extract_from_grid_search(args.results_path, args.label_to_extract)
