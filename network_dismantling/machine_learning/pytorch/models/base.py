from argparse import ArgumentParser

from torch.nn import Module


class BaseModel(Module):

    _model_parameters = []
    _affected_by_seed = False

    @staticmethod
    def add_model_parameters(parser: ArgumentParser):
        pass

    @staticmethod
    def parameters_callback(args):
        pass

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def get_parameters(cls):
        return cls._model_parameters

    @classmethod
    def is_affected_by_seed(cls):
        return cls._affected_by_seed

    def set_seed(self, seed):
        pass

    @staticmethod
    def parameters_combination_validator(params):
        return params

    def add_run_parameters(self, run: dict):
        pass

    def model_name(self):
        pass
