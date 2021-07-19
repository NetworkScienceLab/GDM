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

from argparse import ArgumentParser
from ast import literal_eval
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import NullLocator, MaxNLocator
from pathlib2 import Path
from sklearn.preprocessing import minmax_scale

from network_dismantling.machine_learning.pytorch.grid_output import prepare_df, filtered_columns, replace_labels, \
    rw_early_warning_networks

sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks")
# sns.set_context("paper")
sns.set_context("talk")


rw_early_warning_networks


def load_and_display_df(args):
    df = pd.read_csv(args.file)

    display_df(df, args)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def display_df(df, args, print=print):
    prepare_df(df, args)

    if args.heuristics_file:
        dfh = pd.read_csv(args.heuristics_file)

        if args.plot_query is not None:
            dfh.query(args.plot_query, inplace=True)

    groups = df.groupby(["network"])

    for network_name, group_df in groups:
        group_df_head = group_df.head(args.show_first)
        group_df_head_filtered = group_df_head.loc[:, [x for x in group_df_head.columns if
                                                       x not in filtered_columns]]
        print("Network {}, showing first {}/{} runs:\n{}\n".format(network_name,
                                                                   min(args.show_first, group_df.shape[0]),
                                                                   group_df.shape[0],
                                                                   group_df_head_filtered.set_index("idx")
                                                                   )
              )

        group_df = group_df_head
        if (group_df.shape[0] == 1) \
                and (args.plot or args.verbose):
            group_df.reset_index(drop=True, inplace=True)
            infos = group_df.loc[0, :]
            removals = literal_eval(infos.pop("removals"))

            print(infos[df.columns.difference(["network"])])

            heuristic_name = "dynamic" if infos["static"] == False else ""  # "static"

            if "_reinserted" in Path(args.file).stem:
                heuristic_name += " machine learning + R"
            else:
                heuristic_name += " machine learning"

            print("Dismantling the network according to {} (aiming to LCC size {})".format(heuristic_name.upper(), ""))

            num_removals = len(removals)

            if not args.plot and args.verbose:
                for removal in removals:
                    print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

            else:
                removals_list = []
                if args.heuristics_file:
                    dfh_filtered = dfh.loc[(dfh["network"] == network_name), :]

                    print(dfh_filtered)

                    for _, row in dfh_filtered.iterrows():
                        removals_list.append((
                            # "static"
                            ("" if row["static"] else "dynamic") + " " + row["heuristic"].replace("_", " "),
                            literal_eval(row["removals"])
                        ))

                pis_removals = removals
                pis_vertices = list(map(itemgetter(1), pis_removals))
                pis_values = np.array(list(map(itemgetter(2), pis_removals)))

                # Apply thresholding
                removals_slcc = np.array(list(map(itemgetter(4), pis_removals)))
                removals_slcc_max_idx = removals_slcc.argmax()
                pis_threshold = pis_values[removals_slcc_max_idx]

                cap = np.sum(pis_values[pis_values >= pis_threshold])

                pis_values = minmax_scale(pis_values)

                pis_dict = dict(zip(pis_vertices, list(pis_values)))
                pis_dict = defaultdict(lambda: 0, pis_dict)

                output_folder = args.plot_output / "{}_{}".format(network_name, infos["idx"])

                for heuristic_name, heuristic_removals in removals_list:
                    base_heuristic_name = str(heuristic_name.strip())

                    heuristic_name = base_heuristic_name.title()
                    heuristic_name = replace_labels.get(heuristic_name, heuristic_name)

                    # Create new figure
                    fig, ax1 = plt.subplots()
                    plt.xticks(rotation=45)

                    lcc_axis = ax1
                    area_axis = lcc_axis
                    delta_delta_pi_axis = lcc_axis.twinx()
                    delta_pi_axis = lcc_axis.twinx()
                    pi_axis = lcc_axis.twinx()

                    lcc_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                    delta_pi_axis.yaxis.set_major_locator(NullLocator())
                    delta_delta_pi_axis.yaxis.set_major_locator(NullLocator())
                    pi_axis.tick_params(axis='y')

                    lines = []

                    num_removals = len(heuristic_removals)

                    x = list(map(itemgetter(0), heuristic_removals))
                    y = list(map(itemgetter(3), heuristic_removals))

                    vertices = np.array(list(map(itemgetter(1), heuristic_removals)))
                    pis = [pis_dict[x] for x in vertices]

                    slcc = np.array(list(map(itemgetter(4), heuristic_removals)))
                    peak_slcc = np.argmax(slcc)

                    pi = np.cumsum(pis)
                    pi[pi > cap] = cap

                    delta_pi = pis

                    lines.extend(
                        lcc_axis.plot(x, y, '-o', markersize=4, linewidth=2, color="#084488",
                                      label="LCC", zorder=100)
                    )

                    peak_slcc = int(peak_slcc)
                    area_axis.fill_between(x[:peak_slcc], [max(y)] * len(x[:peak_slcc]), color="#2ca02c", alpha=0.2, zorder=-1)
                    area_axis.fill_between(x[peak_slcc-1:], [max(y)] * len(x[peak_slcc-1:]), color="#d62728", alpha=0.2, zorder=-1)

                    lines.extend(
                        lcc_axis.plot(x, slcc, '--s', markersize=2, linewidth=1, color="#3080bd", #alpha=0.7,
                                      label="SLCC", zorder=50)
                    )

                    lines.extend(
                        pi_axis.plot(x, pi / cap, '-', markersize=0, linewidth=2,
                                     color="#d62728",
                                     # color="#d62728",
                                     label=r'$\Omega$',
                                     zorder=110,
                                     )
                    )

                    lines.extend(
                        delta_pi_axis.plot(x, delta_pi, '--', markersize=0, linewidth=0.5,
                                           color="#ff7f0e",
                                           label="PI",
                                           zorder=40,
                                           alpha=0.7,
                                           )
                    )

                    delta_pi_axis.set_yticklabels([])

                    delta_delta_pi_axis.set_yticklabels([])

                    lcc_axis.set_xbound(lower=1)

                    # Add labels
                    if "roads" in network_name:
                        lcc_axis.set_xlabel('# of attacked intersections')
                    elif "london" in network_name or \
                            "grid" in network_name:
                        lcc_axis.set_xlabel('# of attacked stations')
                    else:
                        lcc_axis.set_xlabel('# of attacked nodes')

                    lcc_axis.set_ylabel('Robustness')
                    pi_axis.set_ylabel('$\Omega$ value')

                    # Despine the plot
                    sns.despine(left=True, right=True)

                    plt.tight_layout()

                    if args.plot_output is None:
                        # plt.xlim(right=num_removals * (1.10))
                        plt.show()
                    else:
                        xlim_right = num_removals * 1.02

                        plt.xlim(right=xlim_right)

                        file = output_folder / ("{}.pdf".format(base_heuristic_name))

                        if not file.parent.exists():
                            file.parent.mkdir(parents=True)

                        plt.savefig(str(file), bbox_inches='tight')

                        file = file.with_suffix(".removals.csv")
                        with file.open("w+") as f:
                            f.write("node,lcc,slcc,ew\n")
                            for i, v in enumerate(vertices):
                                f.write("{},{},{},{}\n".format(v, y[i], slcc[i], pi[i]/cap))

                        # plt.clean()
                        plt.close()

                    # file = output_folder / "pis.csv"
                    # with file.open("w+") as f:
                    #     f.write("node,pi\n")
                    #     for v, pi in pis_dict.items():
                    #         f.write("{},{}\n".format(v, pi))
        else:
            exit("You should provide just one run to get the probabilities. {} found.".format(group_df.shape[0]))


FUNCTION_MAP = {
    'display_df': load_and_display_df,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description=""
    )

    parser.add_argument(
        '--command',
        type=str,
        choices=FUNCTION_MAP.keys(),
        default="display_df",
        required=False
    )

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        required=True,
        help="Output DataFrame file location",
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
        "-sf",
        "--show_first",
        type=int,
        default=15,
        required=False,
        help="Show first N dismantligs",
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        required=False,
        action="store_true",
        help="Plot the (single) result and the heuristics on the same network",
    )
    parser.add_argument(
        "-po",
        "--plot_output",
        type=Path,
        default=None,
        required=False,
        help="Output plot location",
    )
    parser.add_argument(
        "-fh",
        "--heuristics_file",
        type=Path,
        default="./out/df/heuristics.csv",
        required=False,
        help="Heuristics output DataFrame file location",
    )
    # parser.add_argument(
    #     "-fpi",
    #     "--pi_file",
    #     type=Path,
    #     required=True,
    #     help="PIs output DataFrame file location",
    # )
    parser.add_argument(
        "-qp",
        "--plot_query",
        type=str,
        default=None,
        required=False,
        help="Query the dataframe",
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

    if not args.file.is_absolute():
        args.file = args.file.resolve()

    if not args.heuristics_file.is_absolute():
        args.heuristics_file = args.heuristics_file.resolve()

    args.file = str(args.file)
    args.heuristics_file = str(args.heuristics_file)

    FUNCTION_MAP[args.command](args)
