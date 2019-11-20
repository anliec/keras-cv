import json
import os
import subprocess

from to_tf_lite import keras_to_tf_lite

from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


if __name__ == '__main__':
    additional_config = {"Conv": {"use_resnet": False, "use_mobile_net": False},
                         "Residual": {"use_resnet": True, "use_mobile_net": False},
                         "Inverted residual bottleneck": {"use_resnet": False, "use_mobile_net": True}}
    results = {}

    if os.path.isfile("speed_results_filters.json"):
        with open("speed_results_filters.json", 'r') as j:
            results = json.load(j)
    else:
        filters = []
        for a in range(8, 33, 8):
            filters.append((a, a, a, a))
            # for b in range(a, 33, 8):
            #     for c in range(b, 33, 8):
            #         for d in range(c, 33, 8):
            #             filters.append((a, b, c, d))
        for name, ac in additional_config.items():
            vs = []
            fs = []
            mins = []
            avgs = []
            maxs = []
            for f in filters:
                with open("/tmp/config.json", 'w') as j:
                    json.dump({"size_value": (110, 200), "dropout_rate": 0.1, "dropout_strategy": "last",
                               "layers_filters": f, "expansions": (1, 6, 6)}, j)

                keras_to_tf_lite(keras_model_path="/tmp/nofile", out_path="/tmp/tmp.tflite", config_file="/tmp/config.json",
                                 data_path=None)

                cmd = "/home/nicolas/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model " \
                      "--graph={} --num_threads=1".format("/tmp/tmp.tflite")
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                output = output.decode("utf-8")
                lines = output.split('\n')
                vals = {e.split('=')[0]: float(e.split('=')[1]) * 1e-6 for e in lines[28].split(" ")}
                vs.append(f[0])
                fs.append(f)
                mins.append(vals['min'])
                maxs.append(vals['max'])
                avgs.append(vals['avg'])
            results[name] = (vs, fs, mins, avgs, maxs)
        with open("speed_results_filters.json", 'w') as j:
            json.dump(results, j)

    for name, (vs, fs, mins, avgs, maxs) in results.items():
        plt.plot(vs, avgs, label=name)
        plt.fill_between(vs, mins, maxs, alpha=0.3)
    plt.xlabel("Filters count")
    plt.ylabel("Run time (s)")
    plt.title("Evolution of the computation speed with filter count")
    plt.legend()
    plt.savefig('graphs/speed_by_filters.png')
    plt.show()

    for name, (vs, fs, mins, avgs, maxs) in results.items():
        avgs_fps = [1 / v for v in avgs]
        mins_fps = [1 / v for v in mins]
        maxs_fps = [1 / v for v in maxs]
        plt.plot(vs, avgs_fps, label=name)
        plt.fill_between(vs, maxs_fps, mins_fps, alpha=0.3)
    plt.xlabel("Filters count")
    plt.ylabel("Frame per second (fps)")
    plt.title("Evolution of the computation speed with the filter count")
    plt.legend()
    plt.savefig('graphs/speed_by_filters_fps.png')
    plt.show()




