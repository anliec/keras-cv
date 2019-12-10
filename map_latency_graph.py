from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

"""
A script that generate graph and tables for the reports from values get from trainings.
"""

data = {
    "Yolov3-tiny": [
        {
            "comment": "Tiny Yolo at 704x416",
            "mAP": {50: 0.92, 25: 0.97},
            "avg_latency": 1.73753
        }, {
            "comment": "Tiny Yolo at 576x320",
            "mAP": {50: 0.96, 25: 0.98},
            "avg_latency": 1.07467
        }, {
            "comment": "Tiny Yolo at 160x288",
            "mAP": {50: 0.7681, 25: 0.934},
            "avg_latency": 0.251755
        }, {
            "comment": "Tiny Yolo at 96x160",
            "mAP": {50: 0.39, 25: 0.7026},
            "avg_latency": 0.0783643
        }
    ],
    "Residual convolution": [
        {
            "comment": "filters: (6,10)",
            "mAP": {50: 0.4788, 25: 0.5884},
            "avg_latency": 8796.85e-6
        }, {
            "comment": "filters: (8,16)",
            "mAP": {50: 0.55, 25: 0.68},
            "avg_latency": 0.0112777
        }, {
            "comment": "filters: (16,24)",
            "mAP": {50: 0.61, 25: 0.73},
            "avg_latency": 0.0204567
        }, {
            "comment": "filters: (32,64)",
            "mAP": {50: 0.67, 25: 0.82},
            "avg_latency": 0.06757149999999999
        },
    ],
    "Inverted residual bottleneck": [
        {
            "comment": "filters: (8,16)",
            "mAP": {50: 0.38, 25: 0.49},
            "avg_latency": 0.025370999999999998
        }
    ]
}


def plot(th=25):
    for model_type, d in data.items():
        x, y = [], []
        for value in d:
            x.append(value["avg_latency"])
            y.append(value["mAP"][th])
        plt.plot(x, y, '+-', label=model_type)

    plt.legend()
    plt.xlabel("Average latency")
    plt.ylabel("mAP@{}".format(th))
    axes = plt.gca()
    axes.set_xlim([0, None])
    axes.set_ylim([0, 1])
    plt.savefig("graphs/map_at_{}_latency_models.png".format(th))
    plt.show()


def plot_fps(th=25):
    for model_type, d in data.items():
        x, y = [], []
        for value in d:
            x.append(1.0 / value["avg_latency"])
            y.append(value["mAP"][th])
        plt.plot(x, y, '+-', label=model_type)

    plt.legend()
    plt.xlabel("Average frame per second (fps)")
    plt.ylabel("mAP@{}".format(th))
    axes = plt.gca()
    axes.set_xlim([0, None])
    axes.set_ylim([0, 1])
    plt.savefig("graphs/map_at_{}_fps_models.png".format(th))
    plt.show()


def generate_table():
    for model_type, d in data.items():
        first_col = "\\multirow{{ {} }}{{*}}{{ {} }}".format(len(d), model_type)
        for value in d:
            print("\t{} & {} & {:.2f} & {:.2f} & {:.3f} & {:.1f} \\\\".format(first_col,
                                                                              value["comment"],
                                                                              value['mAP'][50],
                                                                              value['mAP'][25],
                                                                              value['avg_latency'],
                                                                              1 / value['avg_latency'])
                  )
            first_col = " "
        print("\t\\hline")


if __name__ == '__main__':
    plot(50)
    plot(25)
    plot_fps(50)
    plot_fps(25)
    generate_table()

