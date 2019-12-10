"""
Test a set of model parameters by loading the model with random weight, converting it to tfLite and running it
on a connected android device.
Some path may need to be updated to be used on other system.
See https://www.tensorflow.org/lite/performance/benchmarks on how to build the needed tools.
"""

import json
import os
import subprocess

from to_tf_lite import keras_to_tf_lite

from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


if __name__ == '__main__':
    additional_config = {  # "Conv": {"use_resnet": False, "use_mobile_net": False},
                         "Residual": {"use_resnet": True, "use_mobile_net": False},
                         "Inverted residual bottleneck": {"use_resnet": False, "use_mobile_net": True}}
    results = {}

    print("pushing binary to device...")
    cmd = "adb push /home/nicolas/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model " \
          "/data/local/tmp && adb shell chmod +x /data/local/tmp/benchmark_model"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Binary Pushed to Device!")

    if os.path.isfile("speed_results_filters_and_filter_list.json"):
        with open("speed_results_filters_and_filter_list.json", 'r') as j:
            results = json.load(j)
    else:
        filters = [(8, 8, 16, 16), (16, 16, 24, 24), (32, 32, 64, 64)]
        # for a in range(8, 33, 8):
        #     filters.append((a, a, a, a))
        #     # for b in range(a, 33, 8):
        #     #     for c in range(b, 33, 8):
        #     #         for d in range(c, 33, 8):
        #     #             filters.append((a, b, c, d))
        for name, ac in additional_config.items():
            vs = []
            fs = []
            mins = []
            avgs = []
            maxs = []
            for f in filters:
                config = {"size_value": (110, 200), "dropout_rate": 0.1, "dropout_strategy": "last",
                          "layers_filters": f, "expansions": (1, 6, 6)}

                for k, v in ac.items():
                    config[k] = v

                with open("/tmp/config.json", 'w') as j:
                    json.dump(config, j)

                tmp_tflite_model_path = "/tmp/tmp.tflite"
                keras_to_tf_lite(keras_model_path="/tmp/nofile", out_path=tmp_tflite_model_path,
                                 config_file="/tmp/config.json", data_path=None)

                cmd = "adb push {} /data/local/tmp/tmp.tflite ".format(tmp_tflite_model_path)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                process.communicate()
                cmd = "adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/tmp.tflite --num_threads=1"
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
        with open("speed_results_filters_and_filter_list.json", 'w') as j:
            json.dump(results, j)

    for name, (vs, fs, mins, avgs, maxs) in results.items():
        plt.plot(vs, avgs, label=name)
        plt.fill_between(vs, mins, maxs, alpha=0.3)
    plt.xlabel("Filters count")
    plt.ylabel("Run time (s)")
    plt.title("Evolution of the computation speed with filter count")
    plt.legend()
    plt.savefig('graphs/speed_by_filters_and_filter_list.png')
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
    plt.savefig('graphs/speed_by_filters_and_fps_filter_list.png')
    plt.show()




