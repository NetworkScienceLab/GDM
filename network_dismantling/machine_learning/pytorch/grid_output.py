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
from operator import itemgetter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pylab
from matplotlib.ticker import MaxNLocator
from pathlib2 import Path

sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks")
# sns.set_context("paper")
sns.set_context("talk")

replace_labels = {
    "Machine Learning": "GDM",
    "Machine Learning +R": "GDM +R",
    "Machine Learning + R": "GDM +R",
    "Gndr": "GND +R",
    "Gnd": "GND",
    "Ms": "MS",
    "Msr": "MS +R",
    "Corehd": "CoreHD",
    "Egnd": "EGND",
    "Ei S1": "EI $\sigma_1$",
    "Ei S2": "EI $\sigma_2$",
    # "Collective Influence L1": "Collective Influence $\ell-1$",
    # "Collective Influence L2": "Collective Influence $\ell-2$",
    # "Collective Influence L3": "Collective Influence $\ell-3$",
    "Collective Influence Ell1 (Dynamic)": "CI $\ell-1$",
    "Collective Influence Ell2 (Dynamic)": "CI $\ell-2$",
    "Collective Influence Ell3 (Dynamic)": "CI $\ell-3$",
    "Degree (Dynamic)": "Adaptive degree",
}

colors_mapping = {
    "GDM": "#3080bd",
    "GDM +R": "#084488",
    "GDM AUC": "#3080bd",
    "GDM +R AUC": "#084488",
    "GDM #Removals": "#3080bd",
    "GDM +R #Removals": "#084488",
    "GND +R": "#ff7f0e",
    "GND": "#ffbb78",
    "MS +R": "#2ca02c",
    "MS": "#98df8a",
    # "Betweenness Centrality": "#8c564b",
    # "Degree": "#9467bd",
    # "Collective Influence L3": "#d62728",
    # "Pagerank": "#ff9896"
    "EGND": "#8c564b",
    "Adaptive degree": "#9467bd",
    "CI $\ell-2$": "#d62728",
    "CoreHD": "#ff9896",
    "EI $\sigma_1$": "#bcbd22"
}

# Define run columns to match the runs
run_columns = [
    # "removals",
    "slcc_peak_at",
    "lcc_size_at_peak",
    "slcc_size_at_peak",
    "r_auc",
    # TODO
    "seed",
    "idx",
    "rem_num",
    # "features"
]

rename_networks = {
    "moreno_train_train": "moreno_train",
    "subelj_jdk_jdk": "subelj_jdk",
    "subelj_jung-j_jung-j": "subelj_jung-j",
}

rw_test_networks = [
    "advogato",
    "arenas-meta",
    "ARK201012_LCC",
    "cfinder-google",
    "corruption",
    "dblp-cite",
    "dimacs10-celegansneural",
    "dimacs10-polblogs",
    "econ-wm1",
    "ego-twitter",
    "eu-powergrid",
    "foodweb-baydry",
    "foodweb-baywet",
    "inf-USAir97",
    "internet-topology",
    "librec-ciaodvd-trust",
    "librec-filmtrust-trust",
    "linux",
    "loc-brightkite",
    "maayan-figeys",
    "maayan-foodweb",
    "maayan-Stelzl",
    "maayan-vidal",
    "moreno_crime_projected",
    "moreno_propro",
    "moreno_train_train",
    "munmun_digg_reply_LCC",
    "oregon2_010526",
    "opsahl-openflights",
    "opsahl-powergrid",
    "opsahl-ucsocial",
    "pajek-erdos",
    "p2p-Gnutella06",
    "p2p-Gnutella31",
    "petster-hamster",
    "power-eris1176",
    "route-views",
    "slashdot-threads",
    "slashdot-zoo",
    "web-webbase-2001",
    "web-EPA",
    "wikipedia_link_kn",
    "wikipedia_link_li",
    "subelj_jdk_jdk",
    "subelj_jung-j_jung-j"
]
rw_large_test_networks = [
    "citeseer",
    "com-dblp",
    "digg-friends",
    "douban",
    "email-EuAll",
    "hyves",
    "loc-gowalla",
    "munmun_twitter_social",
    "petster-catdog-household",
    "tech-RL-caida",
    "twitter_LCC",
    "wordnet-words",
]
rw_early_warning_networks = [
    "gridkit-eupowergrid",
    "gridkit-north_america",
    "london_transport_multiplex_aggr",
    "roads-california",
    "roads-northamerica",
    "roads-sanfrancisco",
    "tech-RL-caida",
    "web-Stanford",
    "web-NotreDame",
]

filtered_columns = ["network", "removals", "model_seed", "seed",
                    "learning_rate", "weight_decay", "negative_slope",
                    "bias", "concat", "removals_num", "dropout"]


def load_and_display_df(args):
    if not (args.verbose or args.plot):
        # Read column names from file
        cols = list(pd.read_csv(str(args.file), nrows=1))

        # Use list comprehension to remove the unwanted column in **usecol**
        df = pd.read_csv(str(args.file), usecols=[i for i in cols if i != 'removals'])

    else:

        df = pd.read_csv(str(args.file))

    display_df(df)


def prepare_df(df, args):
    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    sort_by = [args.sort_column]
    if args.sort_column == "r_auc":
        sort_by.append("rem_num")
    elif args.sort_column == "rem_num":
        sort_by.append("r_auc")

    # Sort the dataframe
    df.sort_values(by=sort_by, ascending=(not args.sort_descending), inplace=True)

    df["idx"] = df.index

    if args.query is not None:
        df.query(args.query, inplace=True)

    df.loc[:, "lcc_size_at_peak"] *= 100
    df.loc[:, "slcc_size_at_peak"] *= 100


def display_df(df, print=print):
    prepare_df(df, args)

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

        if args.query is not None:
            dfr.query(args.query, inplace=True)

        if args.sort_column == "average_dmg":
            dfr["average_dmg"] = (1 - dfr["lcc_size_at_peak"]) / dfr["slcc_peak_at"]
        dfr.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)

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

            if "_reinserted" in args.file.stem:
                heuristic_name += "machine learning + R"
            else:
                heuristic_name += "machine learning"

            print("Dismantling the network according to {} (aiming to LCC size {})".format(heuristic_name.upper(), ""))

            num_removals = len(removals)

            heuristic_name = heuristic_name.strip()
            heuristic_name = str(heuristic_name.strip().title())
            heuristic_name = replace_labels.get(heuristic_name, heuristic_name)

            if not args.plot and args.verbose:
                for removal in removals:
                    print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

            else:
                removals_list = [(heuristic_name, removals)]

                if args.reinsertions_file:
                    reinserted_run = dfr.loc[
                                     ((dfr["network"] == network_name) &
                                      (dfr["features"] == infos["features"]) &
                                      (dfr["static"] == infos["static"])), :]

                    if len(reinserted_run) != 0:
                        reinserted_run = reinserted_run.reset_index().loc[0, :]
                        print(reinserted_run)

                        heuristic_name = "dynamic" if reinserted_run["static"] == False else ""  # "static"
                        heuristic_name += " machine learning + R"

                        heuristic_name = heuristic_name.strip().title()
                        heuristic_name = replace_labels.get(heuristic_name, heuristic_name)

                        removals_list.append((
                            heuristic_name,
                            literal_eval(reinserted_run["removals"])
                        ))

                if args.heuristics_file:
                    dfh_filtered = dfh.loc[(dfh["network"] == network_name), :]

                    print(dfh_filtered)

                    for _, row in dfh_filtered.iterrows():
                        heuristic_name = row["heuristic"].replace("_", " ") + ("" if row["static"] else " (dynamic)")
                        heuristic_name = str(heuristic_name.strip().title())
                        heuristic_name = replace_labels.get(heuristic_name, heuristic_name)

                        removals_list.append((
                            heuristic_name,
                            literal_eval(row["removals"])
                        ))

                # Create new figure
                fig, ax = plt.subplots()
                plt.xticks(rotation=45)

                fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                zindex = len(removals_list)
                for heuristic_name, removals in removals_list:
                    heuristic_name = heuristic_name.strip()

                    color = colors_mapping[heuristic_name]

                    x = list(map(itemgetter(0), removals))
                    y = list(map(itemgetter(3), removals))

                    plt.plot(x, y, ('-o' if "+R" not in heuristic_name else '-s'),
                             markersize=4,
                             linewidth=2,
                             color=color,
                             zorder=zindex,
                             label=str(heuristic_name)
                             )
                    zindex -= 1

                fig.gca().set_xbound(lower=1)

                # Add labels
                plt.xlabel('Number of removed nodes')
                plt.ylabel('LCC Size')

                # Add the legend with some customizations.
                # legend = fig.gca().legend(loc='best', ncol=4, shadow=False)
                # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=4, borderaxespad=0., shadow=False)

                # fig.gca().set_title(network_name)

                # Despine the plot
                sns.despine()

                plt.tight_layout()

                if args.plot_output is None:
                    # plt.xlim(right=num_removals * (1.10))
                    plt.show()
                else:
                    plt.xlim(right=num_removals * (1.05))

                    file = args.plot_output / ("{}_{}.pdf".format(network_name, infos["idx"]))

                    if not file.parent.exists():
                        file.parent.mkdir(parents=True)

                    plt.savefig(str(file), bbox_inches='tight')

                    figLegend = pylab.figure()

                    file = args.plot_output / "legend.pdf"
                    # produce a legend for the objects in the other figure
                    pylab.figlegend(*ax.get_legend_handles_labels(), ncol=len(removals_list), loc='upper left')

                    figLegend.savefig(str(file), bbox_inches='tight')

                plt.close('all')


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
    parser.add_argument(
        "-fr",
        "--reinsertions_file",
        default=False,
        action="store_true",
        help="Reinsertions output DataFrame file location",
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
        "-v",
        "--verbose",
        default=False,
        required=False,
        action="store_true",
        help="Show the (single) result and the heuristics on the same network",
    )

    args, cmdline_args = parser.parse_known_args()

    if not args.file.is_absolute():
        args.file = args.file.resolve()

    if not args.heuristics_file.is_absolute():
        args.heuristics_file = args.heuristics_file.resolve()

    args.heuristics_file = str(args.heuristics_file)

    FUNCTION_MAP[args.command](args)
