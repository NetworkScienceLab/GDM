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

import numpy as np
from graph_tool.all import *


def get_lcc_slcc(network):
    # Networks are undirected and this is checked after load phase
    # Forcing directed = False triggers a GraphView call which is expensive
    belongings, counts = label_components(network) #, directed=False)
    counts = counts.astype(int, copy=False)

    if len(counts) == 0:
        local_network_lcc_size, local_network_slcc_size = 0, 0
        lcc_index = 0
    elif len(counts) < 2:
        local_network_lcc_size, local_network_slcc_size = counts[0], 0
        lcc_index = 0
    else:
        lcc_index, slcc_index = np.argpartition(np.negative(counts), 1)[:2]
        local_network_lcc_size, local_network_slcc_size = counts[[lcc_index, slcc_index]]

    return belongings, local_network_lcc_size, local_network_slcc_size, lcc_index


def lcc_threshold_dismantler(network, node_generator, generator_args, stop_condition):
    removals = []

    network.set_fast_edge_removal(fast=True)
    network_size = network.num_vertices()

    # Init generator
    generator = node_generator(network, **generator_args)
    response = None

    # Get static and dynamic vertex IDs
    static_id = network.vertex_properties["static_id"].get_array()
    dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # dynamic_id = np.empty(shape=network_size, dtype=np.int64)
    # for v in network.get_vertices():
    #     dynamic_id[static_id[v]] = network.vertex_index[v]
    #     assert dynamic_id[static_id[v]] == v

    # # Init last valid vertex
    # last_vertex = network_size - 1

    # Compute connected component sizes
    belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

    # Init removals counter
    i = 0
    while True:
        v_i_static, p = generator.send(response)

        # Find the vertex in graphtools and remove it
        v_i_dynamic = dynamic_id[v_i_static]

        if belongings[v_i_dynamic] != lcc_index:
            response = False
        else:
            response = True

            v_gt = network.vertex(v_i_dynamic, use_index=True, add_missing=False)

            # try:
            #     assert static_id[v_i_dynamic] == v_i_static
            #     # assert dynamic_id[static_id[v_i_dynamic]] == v_i_dynamic
            #
            # except Exception as e:
            #     print("ASSERT FAILED: static_id", static_id[v_i_dynamic], "==", "v_i_static", v_i_static)
            #     # print("A2", dynamic_id[static_id[v_i_dynamic]], "==", v_i_dynamic)
            #     raise e

            # dynamic_id[static_id[last_vertex]] = v_i_dynamic
            # network.remove_vertex(v_gt, fast=True)
            # last_vertex -= 1
            network.clear_vertex(v_gt)

            i += 1

            # Compute connected component sizes
            belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

            removals.append(
                (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
            )

        if local_network_lcc_size <= stop_condition:
            generator.close()
            break

    return removals


def lcc_peak_dismantler(network, node_generator, generator_args, stop_condition, logger=print):
    removals = []

    network.set_fast_edge_removal(fast=True)
    network_size = network.num_vertices()

    # Init generator
    generator = node_generator(network, **generator_args)
    response = None

    # Init removals counter
    i = 0

    # Get static and dynamic vertex IDs
    static_id = network.vertex_properties["static_id"].get_array()
    dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # dynamic_id = np.empty(shape=network_size, dtype=np.int64)
    # for v in network.get_vertices():
    #     dynamic_id[static_id[v]] = network.vertex_index[v]
    #     assert dynamic_id[static_id[v]] == v

    # # Init last valid vertex
    last_vertex = network_size - 1

    # Compute connected component sizes
    belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

    # Init peak SLCC value
    peak_network_slcc_size = local_network_slcc_size

    while True:
        v_i_static, p = generator.send(response)

        # Find the vertex in graphtools and remove it
        v_i_dynamic = dynamic_id[v_i_static]

        if belongings[v_i_dynamic] != lcc_index:
            response = False
        else:
            response = True

            v_gt = network.vertex(v_i_dynamic, use_index=True, add_missing=False)

            # try:
            #     assert static_id[v_i_dynamic] == v_i_static
            #     # assert dynamic_id[static_id[v_i_dynamic]] == v_i_dynamic
            #
            # except Exception as e:
            #     print("ASSERT FAILED: static_id", static_id[v_i_dynamic], "==", "v_i_static", v_i_static)
            #     # print("A2", dynamic_id[static_id[v_i_dynamic]], "==", v_i_dynamic)
            #     raise e

            # dynamic_id[static_id[last_vertex]] = v_i_dynamic
            # network.remove_vertex(v_gt, fast=True)
            last_vertex -= 1
            network.clear_vertex(v_gt)

            i += 1

            # Compute connected component sizes
            belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

            if peak_network_slcc_size < local_network_slcc_size:
                peak_network_slcc_size = local_network_slcc_size

            removals.append(
                (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
            )

        if (peak_network_slcc_size >= local_network_lcc_size) or \
                (local_network_lcc_size <= stop_condition):
            break

    # TODO REMOVE ME
    for v, p in generator:
        removals.append(
            (i, v, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
        )

        last_vertex -= 1

        if last_vertex < 0:
            break

    # TODO END REMOVE ME

    generator.close()

    return removals, None, None #prediction_time, dismantle_time
