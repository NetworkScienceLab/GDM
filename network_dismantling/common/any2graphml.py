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

import networkx as nx
from glob2 import glob
from networkx import readwrite
from pathlib2 import Path

_format_mapping = {
    "graphml":      ("read_graphml", "write_graphml"),
    "net":        ("read_pajek", "write_pajek"),
    "pajek":      ("read_pajek", "write_pajek"),
    "adjacency":  ("read_adjlist", "write_adjlist"),
    "adj":        ("read_adjlist", "write_adjlist"),
    "edgelist":   ("read_edgelist", "write_edgelist"),
    "edge":       ("read_edgelist", "write_edgelist"),
    "edges":      ("read_edgelist", "write_edgelist"),
    "el":         ("read_edgelist", "write_edgelist"),
    "pickle":     ("read_gpickle", "write_gpickle"),
    "picklez":    ("read_gpickle", "write_gpickle"),
}


def get_supported_exts():
    return _format_mapping.keys()


def get_io_helpers(file=None, ext=None):
    from pathlib2 import Path

    if file is not None:
        ext = Path(file).suffix[1:]
    elif ext is None:
        raise ValueError("No parameter is provided")

    try:
        methods = _format_mapping[ext]
    except KeyError as e:
        raise ValueError("Format not supported {}".format(e))

    return getattr(readwrite, methods[0]), getattr(readwrite, methods[1])


def main(args):
    if args.output is None:
        args.output = Path(args.input)

    if type(args.ext) is not list:
        args.ext = [args.ext]

    if not args.output.exists():
        args.output.mkdir(parents=True)

    files = []
    for ext in args.ext:
        files.extend(glob(args.input + "*." + ext))

    if args.directed:
        print("Using DiGraph")
        create_using = nx.DiGraph
    else:
        print("Using Graph")
        create_using = nx.Graph

    for file in files:
        print("----------\nfile {}".format(file))

        network = None

        file = Path(file)
        output_file = args.output / (file.stem + ".graphml")

        if output_file.exists():
            print("Skipping file {} as it already exists".format(file.stem))
            continue

        reader, writer = get_io_helpers(ext=file.suffix[1:])

        try:
            network = reader(str(file), create_using=create_using, data=(('weight', float),))
        except TypeError as e:
            print(e)
            try:
                network = reader(str(file), create_using=create_using)
            except TypeError as e:
                print(e)

                try:
                    network = reader(str(file))
                except TypeError as e:
                    exit("Error reading file {}".format(e))

        nx.write_graphml(network, str(output_file))
        # , data=(args.no_weights is True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Location of the input networks (directory)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=False,
        help="Location of the output directory",
    )

    parser.add_argument(
        "-d",
        "--directed",
        default=False,
        action='store_true',
        help="Directed",
    )

    parser.add_argument(
        "-nw",
        "--no_weights",
        default=False,
        action='store_true',
        help="Disgard weights",
    )

    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        nargs="*",
        default=sorted(get_supported_exts()),
        required=False,
        help="Extension without dot",
    )

    args, cmdline_args = parser.parse_known_args()

    main(args)
