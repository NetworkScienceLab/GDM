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

from datetime import timedelta
from operator import itemgetter
from time import time

from network_dismantling.common.external_dismantlers.dismantler import Graph, lccThresholdDismantler, thresholdDismantler


def test_network_callback(network):
    from graph_tool.stats import remove_parallel_edges, remove_self_loops

    network = network.copy()

    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]
    edges = list(map(
        lambda e: (
            static_id[e.source()],
            static_id[e.target()]
        ),
        network.edges()
    ))

    if len(edges) == 0:
        raise RuntimeError

    return Graph(edges)


cache = dict()


def add_dismantling_edges(filename, network):
    cache[filename] = test_network_callback(network)

    return cache[filename]


# def _threshold_dismantler(network, predictions, generator_args, stop_condition, dismantler):
def _threshold_dismantler(network, predictor, generator_args, stop_condition, dismantler):

    logger = generator_args["logger"]

    predictions = predictor(network, **generator_args)
    prediction_time = predictions[1]
    predictions = list(predictions[0])

    # TODO IMPROVE SORTING!
    # Get highest predicted value
    logger("Sorting the predictions...")
    start_time = time()
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)
    logger("Done sorting. Took {}".format(timedelta(seconds=(time() - start_time))))

    network_size = network.num_vertices()

    removal_order = list(map(itemgetter(0), sorted_predictions))

    filename = network.graph_properties["filename"]

    try:
        external_network = cache[filename]
    except:
        external_network = add_dismantling_edges(filename, network)

    external_network = Graph(external_network)

    logger("Invoking the external dismantler.")
    start_time = time()

    try:
        raw_removals = dismantler(external_network, removal_order, stop_condition)
    finally:
        del external_network

    dismantle_time = time() - start_time
    logger("External dismantler returned in {}s".format(timedelta(seconds=(dismantle_time))))

    predictions_dict = dict(predictions)

    removals = []
    for i, (s_id, lcc_size, slcc_size) in enumerate(raw_removals, start=1):
        removals.append(
            (i, s_id, float(predictions_dict[s_id]), lcc_size / network_size, slcc_size / network_size)
        )

    del predictions_dict

    return removals, prediction_time, dismantle_time


def lcc_threshold_dismantler(network, predictor, generator_args, stop_condition):
    return _threshold_dismantler(network, predictor, generator_args, stop_condition, lccThresholdDismantler)


def threshold_dismantler(network, predictor, generator_args, stop_condition):
    return _threshold_dismantler(network, predictor, generator_args, stop_condition, thresholdDismantler)
