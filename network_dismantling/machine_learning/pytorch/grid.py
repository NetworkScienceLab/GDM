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
import threading
from itertools import combinations

import pandas as pd
from pathlib2 import Path
from torch import multiprocessing, cuda
from tqdm import tqdm

from network_dismantling.common.multiprocessing import dataset_writer, logger_thread, apply_async, progressbar_thread
from network_dismantling.machine_learning.pytorch.common import product_dict
from network_dismantling.machine_learning.pytorch.config import all_features, threshold
from network_dismantling.common.config import output_path, base_dataframes_path
from network_dismantling.machine_learning.pytorch.dataset_providers import storage_provider
from network_dismantling.machine_learning.pytorch.models.GAT import GAT_Model
from network_dismantling.machine_learning.pytorch.network_dismantler import get_df_columns, prepare_graph


def process_parameters_wrapper(args, df, nn_model, params_queue, test_networks, train_networks, df_queue, log_queue, iterations_queue):
    from torch.multiprocessing import current_process
    from torch import device
    from network_dismantling.machine_learning.pytorch.network_dismantler import add_run_parameters, train_wrapper, test
    from network_dismantling.common.multiprocessing import clean_up_the_pool
    from _queue import Empty

    current_process = current_process()

    def logger(x):
        log_queue.put(current_process.name + " " + str(x))

    runtime_exceptions = 0
    while True:
        try:
            params = params_queue.get_nowait()
        except Empty:
            break

        if params is None:
            break

        key = '_'.join(params.features)
        params.verbose = args.verbose
        params.output_df_columns = args.output_df_columns
        params.target = args.target
        params.threshold = args.threshold
        params.peak_dismantling = args.peak_dismantling
        params.location_train = args.location_train
        params.device = device(params.device)

        # TODO new data structure instead of dict[key] ?

        # Train or load the model
        try:
            model = train_wrapper(params, nn_model=nn_model, networks_provider=train_networks[key], print=logger)

            if params.device:
                try:
                    _model = model
                    model.to(params.device)
                finally:
                    del _model
        except RuntimeError as e:
            logger("ERROR: {}".format(e))
            runtime_exceptions += 1

            continue

        # TODO improve me
        filter = {}
        add_run_parameters(params, filter, model)
        df_filtered = df.loc[
            (df[list(filter.keys())] == list(filter.values())).all(axis='columns'), ["network", "seed"]]

        for name, network, data in test_networks[key]:
            network_df = df_filtered.loc[(df_filtered["network"] == name)]

            if nn_model.is_affected_by_seed():
                tested_seeds = network_df["seed"].unique()

                seeds_to_test = args.seed_test - tested_seeds

            else:
                if len(network_df) == 0:
                    seeds_to_test = [next(iter(args.seed_test))]
                else:
                    # Nothing to do. Network was already tested (and seed doesn't matter)
                    continue

            for seed_test in seeds_to_test:
                params.seed_test = seed_test

                try:
                    # Test
                    runs_dataframe = test(params, model=model, networks_provider=[(name, network, data), ], print=logger, print_model=False)

                except RuntimeError as e:
                    logger("ERROR: {}".format(e))
                    runtime_exceptions += 1

                    continue

                df_queue.put(runs_dataframe)

                clean_up_the_pool()

        # TODO fix OOM
        del model
        clean_up_the_pool()

        iterations_queue.put(1)

    if runtime_exceptions > 0:
        print("\n\n\nWARNING: Some runs did not complete due to some runtime exception (most likely CUDA OOM)."
              "Try again with lower GPU load.\n\n\n".format(runtime_exceptions))


def main(args, nn_model):
    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    if args.output_file.exists():
        df = pd.read_csv(args.output_file)
    else:
        df = pd.DataFrame(columns=args.output_df_columns)

    del df["removals"]

    # Init network providers
    train_networks = init_network_provider(args.location_train,
                                           features_list=args.features,
                                           targets=args.target
                                           )

    test_networks = init_network_provider(args.location_test,
                                          features_list=args.features,
                                          filter=args.test_filter,
                                          targets=None,
                                          )
    # List the parameters to try
    params_list = list(product_dict(_callback=nn_model.parameters_combination_validator, **parameters_to_try))

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Create the Multiprocessing Manager
    mp_manager = multiprocessing.Manager()
    # mp_manager = multiprocessing

    # Init queues
    log_queue, df_queue, params_queue, iterations_queue = mp_manager.Queue(), mp_manager.Queue(), mp_manager.Queue(), mp_manager.Queue()

    # Create and start the Logger Thread
    lp = threading.Thread(target=logger_thread, args=(log_queue, tqdm.write), daemon=True)
    lp.start()

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(target=dataset_writer, args=(df_queue, args.output_file), daemon=True)
    dp.start()

    # mpl = multiprocessing.log_to_stderr()
    # mpl.setLevel(logging.INFO)

    devices = []
    locks = dict()

    if cuda.is_available() and not args.force_cpu:
        print("Using GPU(s).")
        for device in range(cuda.device_count()):
            device = "cuda:{}".format(device)
            devices.append(device)
            locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)
    else:
        print("Using CPU.")
        device = 'cpu'
        devices.append(device)
        locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)

    # Fill the params queue
    # TODO ANY BETTER WAY?
    for i, params in enumerate(params_list):
        device = devices[i % len(devices)]
        params.device = device
        params.lock = locks[device]
        params_queue.put(params)

    with multiprocessing.Pool(processes=args.jobs, initializer=tqdm.set_lock,
                              initargs=(multiprocessing.Lock(),)) as p, \
            tqdm(total=len(params_list), ascii=True) as pb:

        # Create and start the ProgressBar Thread
        pbt = threading.Thread(target=progressbar_thread, args=(iterations_queue, pb,), daemon=True)
        pbt.start()

        for i in range(args.jobs):

            def handle_error(ex):
                import traceback
                traceback.print_exception(type(ex), ex, ex.__traceback__)

            apply_async(pool=p,
                        func=process_parameters_wrapper,
                        args=(args,
                              df,
                              nn_model,
                              params_queue,
                              test_networks,
                              train_networks,
                              df_queue,
                              log_queue,
                              iterations_queue
                              ),
                        # callback=_callback,
                        error_callback=handle_error
                        )
        p.close()
        p.join()

    # Gracefully close the daemons
    df_queue.put(None)
    log_queue.put(None)
    iterations_queue.put(None)

    lp.join()
    dp.join()
    pbt.join()


def init_network_provider(location, targets, features_list, filter="*", callback=None):
    test_networks = storage_provider(location, filter=filter, callback=callback)
    test_networks_names, networks = zip(*test_networks)

    pp_test_networks = {}
    for features in features_list:
        key = '_'.join(features)

        # TODO REMOVE THIS LIST
        pp_test_networks[key] = list(zip(test_networks_names, networks,
                                         map(lambda n: prepare_graph(n, features=features, targets=targets), networks)))

    return pp_test_networks


def parse_parameters(nn_model):
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        nargs="*",
        default=["batch_size", "num_epochs", "learning_rate", "weight_decay", "features"],
        help="The features to use",
        # action="append",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=[32],
        nargs="+",
        help="Batch size for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=[30],
        nargs="+",
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        default=[0.005],
        nargs="+",
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=[1e-3],
        nargs="+",
        help="Weight decay",
    )
    parser.add_argument(
        "-lm",
        "--location_train",
        type=Path,
        default=None,
        required=True,
        help="Location of the dataset (directory)",
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
        type=int,
        default=set(),
        nargs="*",
        help="Pseudo Random Number Generator Seed to use during training",
    )
    parser.add_argument(
        "-St",
        "--seed_test",
        type=int,
        default=set(),
        nargs="*",
        help="Pseudo Random Number Generator Seed to use during tests",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default={0},
        nargs="+",
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
        "-sf",
        "--static_features",
        type=str,
        default=["degree"],
        choices=all_features + ["None"],
        nargs="*",
        help="The features to use",
    )
    parser.add_argument(
        "-mf",
        "--features_min",
        type=int,
        default=1,
        help="The minimum number of features to use",
    )
    parser.add_argument(
        "-Mf",
        "--features_max",
        type=int,
        default=None,
        help="The maximum number of features to use",
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
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs.",
    )
    parser.add_argument(
        "-sa",
        "--simultaneous_access",
        type=int,
        default=float('inf'),
        help="Maximum number of simultaneous predictions on CUDA device.",
    )
    parser.add_argument(
        "-FCPU",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Disables ",
    )

    nn_model.add_model_parameters(parser, grid=True)
    args, cmdline_args = parser.parse_known_args()

    if len(args.seed_train) == 0:
        args.seed_train = args.seed.copy()
    if len(args.seed_test) == 0:
        args.seed_test = args.seed.copy()

    args.static_features = set(args.static_features)
    args.features = set(args.features) - args.static_features
    args.static_features = list(args.static_features)

    if args.features_max is None:
        args.features_max = len(args.static_features) + len(args.features)

    args.features_min -= len(args.static_features)
    args.features_max -= len(args.static_features)

    args.features = [sorted(args.static_features + list(c)) for i in range(args.features_min, args.features_max + 1)
                     for c in combinations(args.features, i)]

    return args


if __name__ == "__main__":
    print("Cuda device count {}".format(cuda.device_count()))

    print("Output folder {}".format(output_path))

    multiprocessing.freeze_support()  # for Windows support

    nn_model = GAT_Model
    args = parse_parameters(nn_model=nn_model)

    base_dataframes_path = base_dataframes_path / args.location_train.parent.name / args.target / "T_{}".format(
        float(args.threshold) if not args.peak_dismantling else "PEAK"
    )

    if not base_dataframes_path.exists():
        base_dataframes_path.mkdir(parents=True)

    if not args.location_test.is_absolute():
        args.location_test = args.location_test.resolve()
    if not args.location_train.is_absolute():
        args.location_train = args.location_train.resolve()

    args.output_df_columns = get_df_columns(nn_model)

    args.output_file = base_dataframes_path / (nn_model.get_name() + "_GRID.csv")
    args.time_output_file = args.output_file.with_suffix(".time.csv")

    print("Output file {}".format(args.output_file))
    print("Time output file {}".format(args.time_output_file))

    main(args, nn_model)
