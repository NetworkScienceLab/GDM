#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import tempfile
from ast import literal_eval
from operator import itemgetter
from os import remove
from os.path import relpath, dirname, realpath
from subprocess import check_output
from time import time

import numpy as np
import pandas as pd
from network_dismantling.common.multiprocessing import null_logger
from pathlib2 import Path
from scipy.integrate import simps
from tqdm import tqdm

from network_dismantling.machine_learning.pytorch.dataset_providers import storage_provider
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import lcc_threshold_dismantler


# Define run columns to match the runs
run_columns = [
    # "removals",
    "slcc_peak_at",
    "lcc_size_at_peak",
    "slcc_size_at_peak",
    "r_auc",
    # TODO
    "seed",
    "average_dmg",
    "rem_num",
    "idx"
]


def get_predictions(network, removals, stop_condition, logger=null_logger):
    start_time = time()

    predictions = reinsert(network, removals, stop_condition)

    time_spent = time() - start_time

    predictions = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], predictions))
    return predictions, time_spent


def static_predictor(network, removals, stop_condition, logger=null_logger):
    predictions = get_predictions(network, removals, stop_condition)

    # Get highest predicted value
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)

    for v, p in sorted_predictions:
        yield v, p


def lcc_static_predictor(network, removals, stop_condition):
    predictions = get_predictions(network, removals, stop_condition)

    # Get highest predicted value
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)

    i = 0
    while True:
        if i >= len(sorted_predictions):
            break

        removed = yield sorted_predictions[i]
        if removed is not False:
            # Vertex was removed, remove it from predictions
            del sorted_predictions[i]

            # ... and start over
            i = 0

        else:
            i += 1

    raise RuntimeError("No more vertices to remove!")


def reinsert(network, removals, stop_condition):
    folder = 'network_dismantling/machine_learning/pytorch/reinsertion/'
    cd_cmd = 'cd {} && '.format(folder)
    config_r_file = "config_r.h"
    reinsertion_strategy = 2

    network_fd, network_path = tempfile.mkstemp()
    broken_ft, broken_path = tempfile.mkstemp()
    output_ft, output_path = tempfile.mkstemp()

    nodes = []
    try:
        static_id = network.vertex_properties["static_id"]
        with open(network_fd, 'w+') as tmp:
            for edge in network.edges():
                tmp.write("{} {}\n".format(
                    static_id[edge.source()],
                    static_id[edge.target()]
                )
                )
            # for edge in network.get_edges():
            #     # TODO STATIC ID?
            #     tmp.write("{} {}\n".format(int(edge[0]) + 1, int(edge[1]) + 1))

        with open(broken_ft, "w+") as tmp:
            for removal in removals:
                tmp.write("{}\n".format(removal))

        cmds = [
            'make clean && make',
            './reinsertion -t {}'.format(
                stop_condition,
            )
        ]

        with open(folder + config_r_file, "w+") as f:
            f.write(("const char* FILE_NET = \"{}\";  // input format of each line: id1 id2\n"
                     "const char* FILE_ID = \"{}\";   // output the id of the removed nodes in order\n"
                     "const char* FILE_ID2 = \"{}\";   // output the id of the removed nodes after reinserting\n"
                     "const int Sort_Strategy = {}; // removing order\n"
                     "                             // 0: keep the original order\n"
                     "                             // 1: ascending order - better strategy for weighted case\n"
                     "                             // 2: descending order - better strategy for unweighted case\n"
                     ).format("../" + relpath(network_path, dirname(realpath(__file__))),
                              "../" + relpath(broken_path, dirname(realpath(__file__))),
                              "../" + relpath(output_path, dirname(realpath(__file__))),
                              reinsertion_strategy
                              )
                    )

        for cmd in cmds:
            try:
                check_output(cd_cmd + cmd, shell=True, text=True)  # , stderr=STDOUT))
            except Exception as e:
                raise RuntimeError("ERROR! {}".format(e))

        with open(output_path, 'r+') as tmp:
            for line in tmp.readlines():
                node = int(line.strip())

                nodes.append(node)

    finally:
        remove(network_path)
        remove(broken_path)
        remove(output_path)

    output = np.zeros(network.num_vertices())

    filtered_removals = []
    removed_removals = []
    for x in removals:
        if x in nodes:
            filtered_removals.append(x)
        else:
            removed_removals.append(x)

    nodes = filtered_removals + removed_removals
    # filtered_removals = [x for x in removals if x in nodes]
    # nodes = filtered_removals
    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[n] = p

    return output


def main(args):
    print = tqdm.write

    # Load the runs dataframe...
    df = pd.read_csv(args.file)

    df_columns = df.columns

    if args.output_file.exists():
        output_df = pd.read_csv(args.output_file)
    else:
        output_df = pd.DataFrame(columns=df.columns)

    # ... and query it
    if args.query is not None:
        df = df.query(args.query)

    # Load the networks
    test_networks = dict(
        storage_provider(args.location_test, filter=args.test_filter)
    )

    # Filter the networks in the folder
    df = df.loc[(df["network"].isin(test_networks.keys()))]

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    # Sort dataframe
    df.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)

    predictor = get_predictions
    dismantler = lcc_threshold_dismantler

    groups = df.groupby(["network"])
    for network_name, network_df in groups:
        network_df = network_df.head(args.reinsert_first)

        runs_iterable = tqdm(network_df.iterrows(), ascii=True)
        runs_iterable.set_description(network_name)

        # runs = []
        for _, run in runs_iterable:
            network = test_networks[network_name]

            removals = literal_eval(run.pop("removals"))

            run.drop(run_columns, inplace=True, errors="ignore")

            run = run.to_dict()

            reinserted_run_df = output_df.loc[
                (output_df[list(run.keys())] == list(run.values())).all(axis='columns'), ["network", "seed"]]

            if len(reinserted_run_df) != 0:
                # Nothing to do. Network was already tested
                continue

            stop_condition = int(np.ceil(removals[-1][3] * network.num_vertices()))
            generator_args = {
                "removals": list(map(itemgetter(1), removals)),
                "stop_condition": stop_condition,
                "logger": print,
            }

            removals, _, _ = dismantler(network.copy(), predictor, generator_args, stop_condition)

            peak_slcc = max(removals, key=itemgetter(4))

            _run = {
                "network": network_name,
                "removals": removals,
                "slcc_peak_at": peak_slcc[0],
                "lcc_size_at_peak": peak_slcc[3],
                "slcc_size_at_peak": peak_slcc[4],
                "r_auc": simps(list(r[3] for r in removals), dx=1),
                "rem_num": len(removals),
            }

            for key, value in _run.items():
                run[key] = value

            # Check if something is wrong with the removals
            if removals[-1][2] == 0:
                for removal in removals:
                    print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

                raise RuntimeError

            # runs.append(run)

            run_df = pd.DataFrame(data=[run], columns=network_df.columns)

            if args.output_file is not None:
                kwargs = {
                    "path_or_buf": Path(args.output_file),
                    "index": False,
                    # header='column_names',
                    "columns": df_columns
                }

                # If dataframe exists append without writing the header
                if kwargs["path_or_buf"].exists():
                    kwargs["mode"] = "a"
                    kwargs["header"] = False

                run_df.to_csv(**kwargs)


def parse_parameters():
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default="out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model_REPRODUCE.csv",
        required=False,
        help="Output DataFrame file location",
    )
    parser.add_argument(
        "-lt",
        "--location_test",
        type=Path,
        default=None,
        required=True,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-Ft",
        "--test_filter",
        type=str,
        default="*",
        required=False,
        help="Test folder filter",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        required=False,
        help="Query the dataframe",
    )
    parser.add_argument(
        "-rf",
        "--reinsert_first",
        type=int,
        default=15,
        required=False,
        help="Show first N dismantligs",
    )
    parser.add_argument(
        "-s",
        "--sort_column",
        type=str,
        default="r_auc",
        required=False,
        help="Column used to sort the entries",
    )
    parser.add_argument(
        "-sa",
        "--sort_descending",
        default=False,
        required=False,
        action="store_true",
        help="Descending sorting",
    )
    args, cmdline_args = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_parameters()

    if not args.location_test.is_absolute():
        args.location_test = args.location_test.resolve()

    suffixes = []

    file_split = args.file.stem.split('.')
    file_name = file_split[0]
    if len(file_split) > 1:
        suffixes.extend(file_split[1:])
    suffixes.append(args.file.suffix.replace(".", ""))

    suffix = '.' + '.'.join(suffixes)

    args.output_file = args.file.with_name(file_name + "_reinserted").with_suffix(suffix)
    print("Output file {}".format(args.output_file))

    main(args)
