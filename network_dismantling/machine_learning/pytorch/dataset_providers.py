from glob import glob

from pathlib2 import Path

from network_dismantling.machine_learning.pytorch.common import load_graph


def storage_provider(location, filter="*", callback=None):
    networks = list()
    for file in sorted(glob(str(location / (filter + ".graphml")))):
        filename = Path(file).stem

        network = load_graph(file)

        assert not network.is_directed()

        network.graph_properties["filename"] = network.new_graph_property("string", filename)

        if callback:
            callback(filename, network)

        networks.append((filename, network))

    return networks

