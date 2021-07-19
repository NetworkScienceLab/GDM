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
from copy import deepcopy
from datetime import timedelta
from operator import itemgetter
from random import seed
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib2 import Path
from progressbar import progressbar
from scipy.integrate import simps
from torch_geometric.data import Data

from network_dismantling.common.dismantlers import lcc_peak_dismantler
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import \
    lcc_threshold_dismantler as external_lcc_threshold_dismantler
from network_dismantling.common.multiprocessing import clean_up_the_pool, null_logger
from network_dismantling.machine_learning.pytorch.config import all_features, threshold, base_models_path
from network_dismantling.machine_learning.pytorch.dataset_providers import storage_provider
from network_dismantling.machine_learning.pytorch.training_data_extractor import training_data_extractor


def prepare_graph(network, features=None, targets=None):
    # Retrieve node features and targets
    # features_property = network.vertex_properties["features"]

    # TODO IMPROVE ME
    if features is None:
        features = all_features

    if "None" in features:
        x = np.ones((network.num_vertices(), 1))
    else:
        x = np.column_stack(
            tuple(
                network.vertex_properties[feature].get_array() for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float)

    if targets is None:
        y = None
    else:
        targets = network.vertex_properties[targets]

        y = targets.get_array().copy()
        y = torch.from_numpy(y).to(torch.float)

    edge_index = np.empty((2, 2 * network.num_edges()), dtype=np.int32)
    i = 0
    for e in network.edges():
        # TODO Can we replace the index here?
        # network.edge_index[e]
        edge_index[:, i] = (network.vertex_index[e.source()], network.vertex_index[e.target()])
        edge_index[:, i + 1] = (network.vertex_index[e.target()], network.vertex_index[e.source()])

        i += 2

    edge_index = torch.from_numpy(edge_index).to(torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def get_predictions(network, model, lock, device=None, data=None, features=None, logger=null_logger):
    logger("Sorting the predictions...")
    start_time = time()

    if data is None:
        data = prepare_graph(network, features=features)

    with lock:
        if device:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)

        try:
            predictions = list(model(data.x, data.edge_index))

        finally:
            # Fix OOM
            del data.x
            del data.edge_index
            del data
            clean_up_the_pool()

    predictions = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], predictions))

    time_spent = time() - start_time
    logger("Done predicting dismantling order. Took {} (including GPU access time, if any)".format(timedelta(seconds=time_spent)))

    return predictions, time_spent


def lcc_static_predictor(network, model, lock, data, features, device, logger=null_logger):
    logger("Predicting dismantling order. ")
    start_time = time()

    predictions, _ = get_predictions(network, model, lock, data=data, features=features, device=device)

    # TODO IMPROVE SORTING!
    # Sort by highest prediction value
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)
    logger("Done predicting dismantling order. Took {} (including GPU access time and sorting)".format(timedelta(seconds=(time() - start_time))))

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


def train(args, model, networks_provider=None, print=print):
    print(model)
    # TODO
    loss_op = torch.nn.MSELoss()

    if args.device:
        _model = model
        try:
            model.to(args.device)
        finally:
            del _model

    torch.manual_seed(args.seed_train)
    np.random.seed(args.seed_train)
    seed(args.seed_train)

    if model.is_affected_by_seed():
        model.set_seed(args.seed_train)

    # Load training networks
    if networks_provider is None:
        networks_provider = init_network_provider(args.location_train, features=args.features, targets=args.target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0

        for i, (_, _, data) in enumerate(networks_provider, start=1):
            # num_graphs = data.num_graphs
            data.batch = None
            data = data.to(args.device)
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item()  # * num_graphs
            loss.backward()
            optimizer.step()

        loss = total_loss / i

        # acc = test(val_loader)
        print('Epoch: {:02d}, Loss: {}, Acc: {:.4f}'.format(epoch, loss, 0.0))


def init_network_provider(location, targets, features=None, filter="*"):
    networks_provider = storage_provider(location, filter=filter)
    network_names, networks = zip(*networks_provider)
    networks_provider = list(
        zip(network_names, networks, map(lambda n: prepare_graph(n, features=features, targets=targets), networks))
    )

    return networks_provider


def test(args, model, networks_provider=None, print_model=True, print=print):
    if print_model:
        print(model)

    torch.manual_seed(args.seed_test)
    np.random.seed(args.seed_test)
    seed(args.seed_test)

    if model.is_affected_by_seed():
        model.set_seed(args.seed_test)

    model.eval()

    if args.peak_dismantling:
        predictor = lcc_static_predictor
        dismantler = lcc_peak_dismantler
    else:
        dismantler = external_lcc_threshold_dismantler
        predictor = get_predictions

    if networks_provider is None:
        networks_provider = init_network_provider(args.location_test, features=args.features, targets=None)

    else:
        networks_provider = deepcopy(networks_provider)

    generator_args = {
        "model": model,
        "features": args.features,
        "device": args.device,
        "lock": args.lock,
        "logger": print,
    }

    with torch.no_grad():

        # Init runs buffer
        runs = []
        for filename, network, data in progressbar(networks_provider, redirect_stdout=False):

            network_size = network.num_vertices()

            generator_args["data"] = data

            # Compute stop condition
            stop_condition = int(np.ceil(network_size * float(args.threshold)))
            print("Dismantling {} according to the predictions. Aiming to LCC size {} ({})".format(filename,
                                                                                                   stop_condition,
                                                                                                   stop_condition / network_size))

            removals, prediction_time, dismantle_time = dismantler(network, predictor, generator_args, stop_condition)

            peak_slcc = max(removals, key=itemgetter(4))

            run = {
                "network": filename,
                "removals": removals,
                "slcc_peak_at": peak_slcc[0],
                "lcc_size_at_peak": peak_slcc[3],
                "slcc_size_at_peak": peak_slcc[4],
                "r_auc": simps(list(r[3] for r in removals), dx=1),
                "rem_num": len(removals),
                "prediction_time": prediction_time,
                "dismantle_time": dismantle_time,
            }
            add_run_parameters(args, run, model)

            runs.append(run)

            if args.verbose > 1:
                print("Percolation at {}: LCC {}, SLCC {}, R {}".format(run["slcc_peak_at"], run["lcc_size_at_peak"],
                                                                        run["slcc_size_at_peak"], run["r_auc"]))

            if args.verbose == 2:
                for removal in run["removals"]:
                    print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

            # Fix OOM
            clean_up_the_pool()

        runs_dataframe = pd.DataFrame(data=runs, columns=args.output_df_columns)

    if args.output_file is not None:

        kwargs = {
            "path_or_buf": Path(args.output_file),
            "index": False,
            # header='column_names'
        }

        # If dataframe exists append without writing the header
        if kwargs["path_or_buf"].exists():
            kwargs["mode"] = "a"
            kwargs["header"] = False

        # TODO REMOVE ME
        # kwargs["path_or_buf"] = kwargs["path_or_buf"].with_suffix(".4.csv")

        runs_dataframe.to_csv(**kwargs)

    return runs_dataframe


def parse_parameters(nn_model):
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    add_arguments(nn_model, parser)

    args, cmdline_args = parser.parse_known_args()
    # args.feature_indices = [indices[x] for x in args.features]

    arguments_processing(args)

    return args


def arguments_processing(args):
    if args.seed_train is None:
        args.seed_train = args.seed
    if args.seed_test is None:
        args.seed_test = args.seed

    args.features = sorted(args.features)


def add_arguments(nn_model, parser):
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=30,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        default=0.005,
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "-lm",
        "--location_train",
        type=Path,
        default=None,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-lt",
        "--location_test",
        type=Path,
        default=None,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        required=True,
        help="The target node property",
    )
    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=float(threshold["test"]),
        required=False,
        help="The target threshold",
    )
    parser.add_argument(
        "-Sm",
        "--seed_train",
        nargs="*",
        type=int,
        default=None,
        help="Pseudo Random Number Generator Seed to use during training",
    )
    parser.add_argument(
        "-St",
        "--seed_test",
        nargs="*",
        type=int,
        default=None,
        help="Pseudo Random Number Generator Seed to use during tests",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=0,
        help="Pseudo Random Number Generator Seed",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=["degree", "clustering_coefficient", "kcore"],
        choices=all_features + ["None"],
        nargs="+",
        help="The features to use",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Output DataFrame file location",
    )
    parser.add_argument(
        "-SD",
        "--static_dismantling",
        default=True,
        action="store_true",
        help="[Test only] Static removal of nodes",
    )
    parser.add_argument(
        "-PD",
        "--peak_dismantling",
        default=False,
        action="store_true",
        help="[Test only] Stops the dimantling when the max SLCC size is larger than the current LCC",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "-FCPU",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Disables ",
    )
    nn_model.add_model_parameters(parser)


def train_wrapper(args, nn_model, train_ne=True, networks_provider=None, print=print):
    torch.manual_seed(args.seed_train)
    np.random.seed(args.seed_train)
    seed(args.seed_train)

    model = nn_model(args)

    print(model)

    model.set_seed(args.seed_train)

    model_name = "F{}_{}_L{}_WD{}_E{}_S{}".format(
        '_'.join(args.features),
        model.model_name(),
        args.learning_rate,
        args.weight_decay,
        args.num_epochs,
        args.seed_train if model.is_affected_by_seed() else None
    )

    if args.verbose == 2:
        print(model_name)

    models_path = base_models_path / args.location_train.parent.name / args.target / model.get_name()
    if not models_path.exists():
        models_path.mkdir(parents=True)

    model_weights_file = models_path / (model_name + ".h5")

    if model_weights_file.is_file():
        model.load_state_dict(torch.load(str(model_weights_file)))

    elif train_ne:
        # Init model
        train(args, model, networks_provider, print=print)
        torch.save(model.state_dict(), str(model_weights_file))
    else:
        raise RuntimeError("Model {} not found".format(model_weights_file))
    return model


def get_df_columns(nn_model):
    # "model",
    return ["network", "features", "slcc_peak_at", "lcc_size_at_peak",
            "slcc_size_at_peak", "removals", "static", "removals_num"] + \
           nn_model.get_parameters() + ["model_seed", "num_epochs", "learning_rate", "weight_decay", "seed", "r_auc", "rem_num", "prediction_time", "dismantle_time"]


def add_run_parameters(params, run, model):
    run["learning_rate"] = params.learning_rate
    run["weight_decay"] = params.weight_decay
    run["num_epochs"] = params.num_epochs
    run["static"] = params.static_dismantling
    run["model_seed"] = params.seed_train
    run["features"] = ','.join(params.features)
    # run["seed"] = params.seed_test

    model.add_run_parameters(run)
