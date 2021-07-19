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
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab
from matplotlib.colors import ListedColormap, rgb2hex
from pathlib2 import Path

from network_dismantling.machine_learning.pytorch.grid_output import replace_labels, rw_large_test_networks, \
    rw_test_networks, rename_networks

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1)

rw_test_networks
rw_large_test_networks


def load_and_display_df(args):
    # Read column names from file
    cols = list(pd.read_csv(str(args.file), nrows=1))

    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(str(args.file), usecols=[i for i in cols if i != 'removals'])

    display_df(args, df)


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

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]
        dfh["average_dmg"] = (1 - dfh["lcc_size_at_peak"]) / dfh["slcc_peak_at"]
        dfr["average_dmg"] = (1 - dfr["lcc_size_at_peak"]) / dfr["slcc_peak_at"]

    # Filter DFs
    columns = ["network", args.sort_column, "static"]
    dfr_columns = columns
    df_columns = columns

    columns = ["network", "heuristic", args.sort_column, "static"]
    dfh = dfh.loc[:, columns]

    if args.query is not None:
        df.query(args.query, inplace=True)
        dfr.query(args.query, inplace=True)

    if not args.large:
        df = df.loc[(df["network"].isin(rw_test_networks))]
    else:
        df = df.loc[(df["network"].isin(rw_large_test_networks))]

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
        group_df.reset_index(inplace=True)
        infos = group_df.iloc[0]

        if "_reinserted" in Path(args.file).stem:
            infos["heuristic"] = "machine learning + R"
        else:
            infos["heuristic"] = "machine learning"

        runs_buffer.append(list(infos[columns].values))

        if args.reinsertions_file:
            try:
                group_dfr = reintroduction_dict[network_name]
                group_dfr.reset_index(inplace=True)

                infos = group_dfr.iloc[0, :]

                infos["heuristic"] = "machine_learning_+R"
                runs_buffer.append(list(infos[columns].values))
            except Exception as e:
                # print(e)
                pass

    runs = pd.DataFrame(data=runs_buffer, columns=columns)
    runs = pd.concat([runs, dfh])

    if not args.large:
        runs = runs.loc[(runs["network"].isin(rw_test_networks))]
        color_palette = sns.color_palette("tab20", n_colors=20) + \
                        sns.color_palette("Set3", n_colors=10) + \
                        sns.color_palette("tab20b", n_colors=5) + \
                        sns.color_palette("tab20", n_colors=10, desat=.5)
    else:
        runs = runs.loc[(runs["network"].isin(rw_large_test_networks))]
        color_palette = sns.color_palette("tab20", n_colors=20)

    runs.reset_index(inplace=True, drop=True)

    runs.loc[runs["static"] == False, "heuristic"] = runs["heuristic"] + " (dynamic)"

    runs["heuristic"] = runs["heuristic"].apply(lambda x: x.replace("_", " ").strip().title())
    runs["network"] = runs["network"].apply(lambda x: rename_networks.get(x, x))
    runs.replace({"heuristic": replace_labels}, inplace=True)

    runs.columns = [x.title() for x in runs.columns]

    cmap = ListedColormap([rgb2hex(x) for x in color_palette])

    runs = runs.pivot_table(index=args.index.title(), columns=args.columns.title(), values=args.sort_column.title())

    if args.index == "Network":
        runs = runs.div(runs.loc[:, "GDM"], axis="index")
        df_sum = runs.mean(axis=0).rename('Average')

        runs = runs.reindex(reorder_heuristics(df_sum, args.reinsertions_file), axis=1)
        output_df = runs.append(df_sum)  # , ignore_index=True)

    else:
        runs = runs.div(runs.loc["GDM", :])

        df_sum = runs.mean(axis=1).rename('Average')

        runs = runs.reindex(index=reorder_heuristics(df_sum, args.reinsertions_file))
        output_df = pd.concat([runs, df_sum], axis=1)

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
        legend=False,
    )

    fig.axes.yaxis.grid(True)
    fig.axes.xaxis.grid(False)

    fig.axes.set_yticklabels([])

    sns.despine(left=True)

    plt.tight_layout()

    if args.plot_output:
        file = args.plot_output / (
            "barplot_i{}_c{}_q{}_qp{}.pdf".format(args.index, args.columns, args.query, args.plot_query, ))

        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        plt.savefig(str(file), bbox_inches='tight')

        file = file.with_suffix(".csv")
        output_df.to_csv(str(file), sep=',', index=True, header=True)

        file = file.with_suffix(".tex")
        output_df.to_latex(str(file), index=True, header=True, sparsify=True)  # , float_format="%.0f")

        # create a second figure for the legend
        figLegend = pylab.figure()

        file = file.with_suffix(".legend.pdf")
        # produce a legend for the objects in the other figure
        pylab.figlegend(*fig.get_legend_handles_labels(), loc='upper left')

        figLegend.savefig(str(file), bbox_inches='tight')
    else:
        plt.show()


def reorder_heuristics(df_sum, reinsertions):
    sum_values = df_sum.to_dict()
    order = list(map(itemgetter(0), sorted(sum_values.items(), key=itemgetter(1))))

    return ["GDM"] + \
           [x for x in order if ("GDM" not in x and not ("+R" in x or "CoreHD" in x or "CI" in x))] + \
           (["GDM +R"] if reinsertions else []) + \
           [x for x in order if ("GDM" not in x and ("+R" in x or "CoreHD" in x or "CI" in x))]


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
        "-L",
        "--large",
        default=False,
        required=False,
        action="store_true",
        help="Plot the Large Networks results",
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
        default="out/df/heuristics.csv",
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
