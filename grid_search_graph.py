import os
import glob
import json
import numpy as np
import pandas as pd
import sqlite3 as sql
from collections import defaultdict

import matplotlib.pyplot as plt


def generate_sql_z_query(z_name: str, filters: dict):
    where_close = " ( \"{}]\" IS NOT NULL ) ".format(z_name)
    for k, lf in filters.items():
        where_close += "AND ("
        first = True
        for v in lf:
            if not first:
                where_close += " OR "
            else:
                first = False
            where_close += "\"{}\"='{}'".format(k, v)
        where_close += ") "
    query = "SELECT DISTINCT \"{}\" FROM glob_stats".format(z_name)
    if where_close != "":
        query += " WHERE " + where_close
    return query


def generate_sql_xy_query(x_name: str, y_name: str, z_name: str, z_value, filters: dict, x_limit):
    where_close = "\"{}\"='{}'".format(z_name, z_value)
    where_close += "AND \"{0}\" IS NOT NULL ".format(x_name)   # AND \"{0}\" <> 'NaN'
    where_close += "AND \"{0}\" IS NOT NULL ".format(y_name)
    for k, lf in filters.items():
        where_close += "AND ("
        first = True
        for v in lf:
            if not first:
                where_close += " OR "
            else:
                first = False
            where_close += "\"{}\"='{}'".format(k, v)
        where_close += ") "
    if x_limit > 0:
        where_close += "AND " + x_name + " < " + str(x_limit) + " "
    query = "SELECT \"{0}\", min(\"{1}\") as min, max(\"{1}\") as max, median(\"{1}\") as median, avg(\"{1}\") as mean " \
            "FROM glob_stats ".format(x_name, y_name)

    if where_close != "":
        query += "WHERE " + where_close
    query += "GROUP BY " + x_name
    return query


def consolidate_data_from_db(con, x_name: str, y_name: str, z_name: str, filters: dict, x_limit=-1):
    z_value_df = pd.read_sql_query(generate_sql_z_query(z_name, filters), con)
    cd = dict()
    for i in z_value_df.get(z_name):
        query = generate_sql_xy_query(x_name, y_name, z_name, i, filters, x_limit)
        print("Selecting data for", z_name, "=", i)
        cd[i] = pd.read_sql_query(query, con)
    return cd


def generate_plot(con, x_arg_name, y_arg_name, z_arg_name, filters, save_to_file=True, x_limit=-1, x_name=None,
                  y_name=None, y_lim=None):
    if x_name is None:
        x_name = x_arg_name
    if y_name is None:
        y_name = y_arg_name

    cd = consolidate_data_from_db(con, x_arg_name, y_arg_name, z_arg_name, filters, x_limit)

    plt.figure()
    plt.axes()
    for a, d in cd.items():
        plt.plot(d.get(x_arg_name), d.get('median'),
                 label=z_arg_name.replace('_', ' ') + ' = ' + str(a).replace('_', ' '))
        # plt.fill_between(d.get(x_arg_name), d.get('mean') - d.get('std'), d.get('mean') + d.get('std'), alpha=0.3)
        plt.fill_between(d.get(x_arg_name), d.get('min'), d.get('max'), alpha=0.3)
    if y_arg_name == "val_categorical_accuracy":
        plt.ylabel("Validation accuracy")
    else:
        plt.ylabel(y_name.replace('_', ' '))
    plt.xlabel(x_name.replace('_', ' '))
    plt.legend()
    if y_lim is not None:
        plt.ylim(y_lim)
    # plt.title("Evolution of acuracy acording to the number of training examples")
    if save_to_file:
        block_name = "all_"
        if "block_type" in filters.keys():
            block_name = '_'.join(map(str, filters["block_type"])) + '_'
        layers = ""
        if "layers_filters" in filters.keys():
            layers = "_" + '_'.join(map(str, filters["layers_filters"]))
        suffix = ""
        if y_lim is not None:
            suffix += "_zoomed_{}_{}".format(y_lim[0], y_lim[1])
        plt.savefig("graphs/" + block_name + x_name + "_" + y_name + "_" + z_arg_name + layers + suffix + ".png")
    plt.show()


def plot_graph_search(base_dir: str = "grid_search"):
    results = glob.glob(os.path.join(base_dir, "*", "results.json"))

    data = []
    for i, r in enumerate(sorted(results)):
        with open(r, 'r') as f:
            j = json.load(f)
        j["nn_fps"] = np.mean(j["nn_fps"])
        for k, v in j["config"]:
            j[k] = v
        del j["config"]

        maps = defaultdict(int)
        for e, m in j['mAPs']:
            for th, v in m.items():
                maps[th] = max(maps[th], v)
        del j['mAPs']
        for th, v in maps.items():
            j["mAP@{}".format(th)] = v

        stats = {"TP": {}, "FP": {}, "FN": {}}
        tp, fp, fn = j["stats"][-1]
        for (th, tpv), fpv, fnv in zip(tp.items(), fp.values(), fn.values()):
            stats["TP"][th] = tpv
            stats["FP"][th] = fpv
            stats["FN"][th] = fnv
        del j['stats']
        for name, values in stats.items():
            for th, v in values.items():
                j["{}@{}".format(name, th)] = v

        j["test"] = i
        data.append(j.values())

    df = pd.DataFrame(data, columns=j.keys())

    con = sql.Connection(":memory:")

    df.to_sql("glob_stats", con)

    print(len(df))





















