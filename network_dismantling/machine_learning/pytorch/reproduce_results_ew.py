from torch import multiprocessing, cuda

from network_dismantling.common.config import output_path
from network_dismantling.machine_learning.pytorch.network_dismantler import get_df_columns
from network_dismantling.machine_learning.pytorch.reproduce_results import main, nn_model, parse_parameters
from parse import compile


folder_structure_regex = compile("{}/{train_dataset:w}/t_{train_target:f}/{test_target}/{}")


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
    default_file = "out/df/synth_train_NEW/t_0.18/EW/GAT_Model.csv"
    args = parse_reproduce_parameters(default_file)

    args.static_dismantling = True
    args.peak_dismantling = True
    args.threshold = 0.1

    main(args, nn_model)
