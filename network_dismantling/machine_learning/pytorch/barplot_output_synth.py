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
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, rgb2hex
from parse import compile
from pathlib2 import Path

from network_dismantling.common.humanize import intword
from network_dismantling.machine_learning.pytorch.grid_output import replace_labels

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper")

name_regex = compile("{type}_n{num_nodes:d}_{}_{instance_num:d}")

blues = sns.color_palette("Blues", 4)
reds = sns.color_palette("Reds", 4)
greens = sns.color_palette("Greens", 4)
color_dictionary = {
    'Erdos_Renyi_1000': blues[1],
    'Erdos_Renyi_10000': blues[2],
    'Erdos_Renyi_100000': blues[3],
    'planted_sbm_1000': reds[1],
    'planted_sbm_10000': reds[2],
    'planted_sbm_100000': reds[3],
    'static_power_law_1000': greens[1],
    'static_power_law_10000': greens[2],
    'static_power_law_100000': greens[3],
}

colors = blues[1:] + reds[1:] + greens[1:]

names_dict = {
    "Erdos_Renyi": "ER",
    "planted_sbm": "SBM",
    "static_power_law": "CM"
}


def load_and_display_df(args):
    # Read column names from file
    cols = list(pd.read_csv(str(args.file), nrows=1))

    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(str(args.file), usecols=[i for i in cols if i != 'removals'])

    display_df(args, df)


def prettify_network_name(x):
    x = x["network"]
    info = name_regex.parse(x)

    name = "{} ({})".format(
        names_dict[info["type"]],
        intword(info["num_nodes"]),
    )

    return name, info["type"], info["num_nodes"], info["instance_num"]


def display_df(args, df, print=print):
    if args.reinsertions_file:
        suffixes = []

        file_split = args.file.stem.split('.')
        file_name = file_split[0]
        if len(file_split) > 1:
            suffixes.extend(file_split[1:])
        suffixes.append(args.file.suffix.replace(".", ""))

        suffix = '.' + '.'.join(suffixes)

        reinsertions_file = args.file.with_name(file_name + "_reinserted").with_suffix(suffix)

        dfr = pd.read_csv(str(reinsertions_file))
    else:
        dfr = pd.DataFrame(columns=df.columns)

    # Load Heuristics
    dfh = pd.read_csv(str(args.heuristics_file))

    if args.plot_query is not None:
        dfh = dfh.query(args.plot_query)

    dfh = dfh.loc[~(dfh["heuristic"].isin(
        ["kcore_decomposition", "local_clustering_coefficient", "chi_degree", "eigenvector_centrality",
         "collective_influence_l2"]))]

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]
        dfh["average_dmg"] = (1 - dfh["lcc_size_at_peak"]) / dfh["slcc_peak_at"]
        dfr["average_dmg"] = (1 - dfr["lcc_size_at_peak"]) / dfr["slcc_peak_at"]

    # Query the DFs
    if args.query is not None:
        df.query(args.query, inplace=True)
        dfr.query(args.query, inplace=True)

    # Filter DFs
    columns = ["network", args.sort_column, "static"]
    dfr = dfr.loc[:, columns]

    columns = ["network", "heuristic", args.sort_column, "static"]
    dfh = dfh.loc[:, columns]
    df = df.loc[:, columns]

    # Sort DF
    df.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)
    dfr.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)

    # Get groups
    df_groups = df.groupby(["network"])
    dfr_groups = dfr.groupby(["network"])

    if len(dfr_groups):
        reintroduction_dict = dict(list(dfr_groups))
    else:
        reintroduction_dict = {}

    reintroduction_dict = defaultdict(lambda: pd.DataFrame(columns=df.columns), **reintroduction_dict)

    runs_buffer = []
    for network_name, group_df in df_groups:
        # group_df.drop_duplicates(inplace=True)

        group_df.reset_index(inplace=True)
        infos = group_df.iloc[0]

        infos["heuristic"] = "machine_learning"
        runs_buffer.append(list(infos[columns].values))

        if args.reinsertions_file:
            try:
                group_dfr = reintroduction_dict[network_name]

                group_dfr.reset_index(inplace=True)
                infos = group_dfr.iloc[0]

                infos["heuristic"] = "machine_learning_+R"
                runs_buffer.append(list(infos[columns].values))
            except:
                pass

    runs = pd.DataFrame(data=runs_buffer, columns=columns)
    runs = pd.concat([runs, dfh])

    runs.reset_index(inplace=True, drop=True)

    runs[["network", "type", "num_nodes", "instance_num"]] = runs.apply(prettify_network_name, axis=1,
                                                                        result_type="expand")

    print(runs)

    runs["color"] = runs[["type", "num_nodes"]].apply(lambda x: "{}_{}".format(x["type"], x["num_nodes"]), axis=1)
    runs["color"] = runs["color"].apply(lambda x: color_dictionary.get(x, x))

    runs.loc[runs["static"] == False, "heuristic"] = runs["heuristic"] + " (dynamic)"

    runs["heuristic"] = runs["heuristic"].apply(lambda x: x.replace("_", " ").strip().title())
    runs.replace({"heuristic": replace_labels}, inplace=True)

    runs.columns = [x.title() for x in runs.columns]

    # Init rainbow
    cmap = ListedColormap([rgb2hex(x) for x in colors])

    runs = runs.pivot_table(index=args.index.title(),
                            columns=args.columns.title(),
                            values=args.sort_column.title(),
                            aggfunc=np.mean
                            )

    if args.index == "Network":
        runs = runs.reindex(reorder_heuristics(runs.columns, args.reinsertions_file), axis=1)
        runs = runs.div(runs.loc[:, "GDM"], axis="index")
        df_sum = runs.sum(axis=0).div(runs.shape[0], axis="index").rename('Average')

        output_df = runs.append(df_sum)  # , ignore_index=True)
    else:
        runs = runs.reindex(index=reorder_heuristics(runs.index, args.reinsertions_file))
        runs = runs.div(runs.loc["GDM", :])

        df_sum = runs.sum(axis=1).div(runs.shape[1]).rename('Average')
        output_df = pd.concat([runs, df_sum], axis=1)

    # print(runs)
    output_df *= 100
    output_df = output_df.round(1)

    df_sum *= 100
    df_sum = df_sum.round(1)

    runs.clip(upper=3, inplace=True)

    print(output_df)
    print(df_sum)

    for column in runs.columns:
        nan_num = np.count_nonzero(np.isnan(runs[column]))
        if nan_num:
            print("{} NaN values in column {}".format(nan_num, column))

    fig = runs.plot.bar(
        stacked=True,
        cmap=cmap,
        grid=True,
        rot=30,
    )

    fig.axes.yaxis.grid(True)
    fig.axes.xaxis.grid(False)

    fig.axes.set_yticklabels([])

    sns.despine(left=True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # fig.set_size_inches
    plt.tight_layout()

    if args.plot_output:
        file = args.plot_output / (
            "barplot_synth_i{}_c{}_q{}_qp{}.pdf".format(args.index, args.columns, args.query, args.plot_query, ))

        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        plt.savefig(str(file), bbox_inches='tight')

        file = file.with_suffix(".csv")
        output_df.to_csv(str(file), sep=',', index=True, header=True)

        file = file.with_suffix(".tex")
        output_df.to_latex(str(file), index=True, header=True, sparsify=True, float_format="%.2f")
    else:
        plt.show()


def reorder_heuristics(index, reinsertions):
    return ["GDM"] + \
           sorted([x for x in index if ("GDM" not in x and not ("+R" in x or "CoreHD" in x))]) + \
           (["GDM +R"] if reinsertions else []) + \
           sorted([x for x in index if ("GDM" not in x and ("+R" in x or "CoreHD" in x))])


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
        "-p",
        "--plot",
        default=False,
        required=False,
        action="store_true",
        help="Plot the results",
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
        default="./out/df/heuristics.SYNTH.csv",
        required=False,
        help="Heuristics output DataFrame file location",
    )

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

    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default="Network",
        required=False,
        help="Column used as X axis",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        default="Heuristic",
        required=False,
        help="Column used as Y axis",
    )

    parser.add_argument(
        "-P",
        "--pivot",
        default=False,
        action="store_true",
        help="Transpose axis",
    )

    parser.add_argument(
        "-fr",
        "--reinsertions_file",
        default=False,
        action="store_true",
        help="Reinsertions output DataFrame file location",
    )

    args, cmdline_args = parser.parse_known_args()

    if not args.file.is_absolute():
        args.file = args.file.resolve()

    if not args.heuristics_file.is_absolute():
        args.heuristics_file = args.heuristics_file.resolve()

    if args.pivot:
        buff = args.index
        args.index = args.columns
        args.columns = buff

    FUNCTION_MAP[args.command](args)
