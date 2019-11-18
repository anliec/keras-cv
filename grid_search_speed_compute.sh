#!/bin/bash

GRID_SEARCH_DIR="$1"

find "$GRID_SEARCH_DIR" -name model.h5 -exec python3 to_tf_lite.py -i {} -o {}.tflite \;
find "$GRID_SEARCH_DIR" -name "*.tflite" -exec bash -c '$HOME/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model --graph=$0 --num_threads=1 > $0.speed' {} \;
find "$GRID_SEARCH_DIR" -name "*.speed" -exec bash -c 'l=$(sed "29q;d" $0) && v=${l##* avg=} && v=${v%% *} && echo $v > $(dirname $0)/avg_lite_speed' {} \;

python3 map_plotter.py "$GRID_SEARCH_DIR" "$GRID_SEARCH_DIR/map.png"
