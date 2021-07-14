from pathlib2 import Path
import dill


def dataset_writer(queue, output_file):
    kwargs = {
        "path_or_buf": Path(output_file),
        "index": False,
        # header='column_names'
    }

    while True:
        record = queue.get()

        if record is None:
            return

        if len(record):
            # TODO DO NOT CHECK EVERY TIME!
            # If dataframe exists append without writing the header
            if kwargs["path_or_buf"].exists():
                kwargs["mode"] = "a"
                kwargs["header"] = False

            record.to_csv(**kwargs)


def logger(record):
    print(record)
    # TODO store to file!


def progressbar_thread(q, progressbar):
    while True:
        record = q.get()

        if record is None:
            return

        progressbar.update()


def logger_thread(q, logger=logger):
    while True:
        record = q.get()

        if record is None:
            return

        logger(str(record))


def tqdm_logger_thread(q, logger=None):
    from tqdm import tqdm

    if logger is None:
        logger = tqdm.write

    while True:
        record = q.get()

        if record is None:
            return
        logger(record)


def run_dill_encoded(payload):
    """
    https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, func, args, callback=None, error_callback=None):
    payload = dill.dumps((func, args))
    return pool.apply_async(run_dill_encoded, (payload,), callback=callback, error_callback=error_callback)


def clean_up_the_pool(*args, **kwargs):
    """https://discuss.pytorch.org/t/pytorch-multiprocessing-cuda-out-of-memory/53417"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def null_logger(record):
    pass