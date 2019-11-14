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
        print("{}:\t{}\t{}".format(name, stats[label_to_extract], stats['config']['layers_filters']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('results_path',
                        type=str)
    parser.add_argument('label_to_extract',
                        type=str)
    args = parser.parse_args()

    extract_from_grid_search(args.results_path, args.label_to_extract)
