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

    if os.path.isfile("speed_results.json"):
        with open("speed_results.json", 'r') as j:
            results = json.load(j)
    else:
        for name, ac in additional_config.items():
            area = []
            shapes = []
            mins = []
            avgs = []
            maxs = []
            for i in range(5, 30):
                shape = int(110 / 10 * i), int(200 / 10 * i)

                with open("/tmp/config.json", 'w') as j:
                    json.dump({"size_value": shape, "dropout_rate": 0.1, "dropout_strategy": "last",
                               "layers_filters": (16, 16, 24, 24), "expansions": (1, 6, 6)}, j)

                keras_to_tf_lite(keras_model_path="/tmp/nofile", out_path="/tmp/tmp.tflite", config_file="/tmp/config.json",
                                 data_path=None)

                cmd = "/home/nicolas/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model " \
                      "--graph={} --num_threads=1".format("/tmp/tmp.tflite")
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                output = output.decode("utf-8")
                # print(output)
                lines = output.split('\n')
                # print("#"*50)
                # print(lines[28])
                # print("#" * 50)
                vals = {e.split('=')[0]: float(e.split('=')[1]) * 1e-6 for e in lines[28].split(" ")}
                # print(vals)
                area.append(shape[0] * shape[1])
                shapes.append(shape)
                mins.append(vals['min'])
                maxs.append(vals['max'])
                avgs.append(vals['avg'])
            results[name] = (area, shapes, mins, avgs, maxs)
        with open("speed_results.json", 'w') as j:
            json.dump(results, j)

    for name, (area, shapes, mins, avgs, maxs) in results.items():
        plt.plot(area, avgs, label=name)
        plt.fill_between(area, mins, maxs, alpha=0.3)
    plt.xlabel("Input area (pixels)")
    plt.ylabel("Run time (s)")
    plt.title("Evolution of the computation speed with the input size")
    plt.legend()
    plt.savefig('graphs/speed_by_nn_size.png')
    plt.show()

    for name, (area, shapes, mins, avgs, maxs) in results.items():
        avgs_fps = [1 / v for v in avgs]
        mins_fps = [1 / v for v in mins]
        maxs_fps = [1 / v for v in maxs]
        plt.plot(area, avgs_fps, label=name)
        plt.fill_between(area, maxs_fps, mins_fps, alpha=0.3)
    plt.xlabel("Input area (pixels)")
    plt.ylabel("Frame per second (fps)")
    plt.title("Evolution of the computation speed with the input size")
    plt.legend()
    plt.savefig('graphs/speed_by_nn_size_fps.png')
    plt.show()




