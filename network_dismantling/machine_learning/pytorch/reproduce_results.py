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
from ast import literal_eval

import pandas as pd
from numpy import isnan
from parse import compile
from pathlib2 import Path
from torch import multiprocessing, cuda
from tqdm import tqdm

from network_dismantling.common.config import output_path
from network_dismantling.common.multiprocessing import dataset_writer, logger_thread, apply_async, progressbar_thread
from network_dismantling.machine_learning.pytorch.common import dotdict
from network_dismantling.machine_learning.pytorch.grid import init_network_provider
from network_dismantling.machine_learning.pytorch.models.GAT import GAT_Model
from network_dismantling.machine_learning.pytorch.network_dismantler import get_df_columns

folder_structure_regex = compile("{}/{train_dataset:w}/t_{train_target:f}/T_{test_target}/{}")

nn_model = GAT_Model


def process_parameters_wrapper(args, df, nn_model, params_queue, test_networks, df_queue, log_queue, iterations_queue):
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
        params.location_train = args.location_train
        params.verbose = args.verbose
        params.static_dismantling = args.static_dismantling
        params.output_df_columns = args.output_df_columns
        params.target = args.target
        params.threshold = args.threshold
        params.peak_dismantling = args.peak_dismantling
        params.device = device(params.device)
        args.seed_test = {params.seed_test}

        # TODO new data structure instead of dict[key] ?

        # Train or load the model
        try:
            model = train_wrapper(params, nn_model=nn_model, networks_provider=None, train_ne=False, print=logger)

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
            # DISMANTLE ONLY SPECIFIC NETWORK
            if name != params.network:
                continue

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
    if args.file.exists():
        df = pd.read_csv(str(args.file))
    else:
        raise RuntimeError("Runs file not found.")

    # REMOVE RUN INFO COLUMNS.
    # NOTE THAT NO INFO ON REMOVALS IS USED AFTER THIS POINT.
    del df["removals"]
    del df["r_auc"]
    del df["rem_num"]
    del df["slcc_peak_at"]
    del df["lcc_size_at_peak"]
    del df["slcc_size_at_peak"]

    if args.output_file.exists():
        reproduce_df = pd.read_csv(args.output_file)
    else:
        reproduce_df = pd.DataFrame(columns=args.output_df_columns)

    test_networks = init_network_provider(args.location_test,
                                          features_list=args.features,
                                          filter=args.test_filter,
                                          targets=None,
                                          )
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

    # Fill dismantling queue
    for i, (_, row) in enumerate(df.iterrows()):
        row = row.to_dict()
        params = dotdict()

        # TODO CHECK IF NETWORK EXISTS FIRST ?
        params.network = row["network"]

        params.learning_rate = row["learning_rate"]
        params.weight_decay = row["weight_decay"]
        params.num_epochs = row["num_epochs"]
        params.static = row["static"]
        params.seed_train = row["model_seed"]
        params.seed_test = row["seed"] if (not isnan(row["seed"])) else row["model_seed"]
        params.features = row["features"].strip().split(",") #args.features[0]

        for parameter in nn_model._model_parameters:
            params[parameter] = [literal_eval(x) for x in row[parameter].strip(",").split(",")]

        # CPU/GPU simultaneous access
        device = devices[i % len(devices)]
        params.device = device
        params.lock = locks[device]

        params_queue.put(params)

    with multiprocessing.Pool(processes=args.jobs, initializer=tqdm.set_lock,
                              initargs=(multiprocessing.Lock(),)) as p, \
            tqdm(total=params_queue.qsize(), ascii=True) as pb:

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
                              reproduce_df,
                              nn_model,
                              params_queue,
                              test_networks,
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


def parse_parameters(default_file):
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=default_file,
        required=False,
        help="Output DataFrame file location",
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
        default=None,
        help="Maximum number of simultaneous predictions on CUDA device.",
    )
    parser.add_argument(
        "-FCPU",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Disables ",
    )

    args, cmdline_args = parser.parse_known_args()

    return args


def get_timedf_columns(nn_model):
    # "model",
    return ["network", "features", "static", "removals_num", "prediction_time", "dismantle_time"] + \
           nn_model.get_parameters() + ["model_seed", "num_epochs", "learning_rate", "weight_decay", "seed", "rem_num"]


def parse_reproduce_parameters(default_file):
    print("Cuda device count {}".format(cuda.device_count()))
    print("Output folder {}".format(output_path))
    multiprocessing.freeze_support()  # for Windows support

    args = parse_parameters(default_file=default_file)

    if args.simultaneous_access is None:
        args.simultaneous_access = args.jobs

    if not args.file.is_absolute():
        args.file = args.file.resolve()
    print("Opening file {}".format(args.file))

    # Setup folders
    try:
        folder_structure = folder_structure_regex.parse(str(args.file))
    except Exception as e:
        raise RuntimeError(
            "ERROR: Expecting input file in folder structure like .../<TRAIN_DATASET_NAME>/<TRAIN_TARGET>/<DISMANTLE_TARGET>/ to match parameters")

    train_set = folder_structure["train_dataset"]  # args.file.parent.parent.parent.name
    threshold = folder_structure["test_target"]  # float(args.file.parent.parent.name.replace("T_", ""))
    train_target = folder_structure["train_target"]  # args.file.parent.name

    args.location_train = (args.location_test.parent / train_set / "dataset").resolve()
    args.threshold = threshold  # args.file.parent.parent.name
    args.target = "t_{}".format(train_target)

    args.features = [["chi_degree", "clustering_coefficient", "degree", "kcore"]]

    if not args.location_test.is_absolute():
        args.location_test = args.location_test.resolve()
    if not args.location_train.is_absolute():
        args.location_train = args.location_train.resolve()

    args.output_df_columns = get_df_columns(nn_model)

    output_suffixes = []

    file_split = args.file.stem.split('.')
    file_name = file_split[0]
    if len(file_split) > 1:
        output_suffixes.extend(file_split[1:])
    output_suffixes.append(args.file.suffix.replace(".", ""))
    output_suffix = '.' + '.'.join(output_suffixes)
    file_name += "_REPRODUCE"

    args.output_file = args.file.with_name(file_name).with_suffix(output_suffix)
    print("Output DF: {}".format(args.output_file))

    return args


if __name__ == "__main__":
    default_file = "out/df/synth_train_NEW/t_0.18/T_0.1/GAT_Model.csv"
    args = parse_reproduce_parameters(default_file)

    args.static_dismantling = True
    args.peak_dismantling = False

    main(args, nn_model)
