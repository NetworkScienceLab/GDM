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

# from collections import defaultdict

import torch
# from models.base import BaseModel
from network_dismantling.machine_learning.pytorch.common import DefaultDict
from network_dismantling.machine_learning.pytorch.models.base import BaseModel
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from common import dotdict


class GAT_Model(BaseModel):
    _model_parameters = ["conv_layers", "heads", "fc_layers", "concat", "negative_slope", "dropout", "bias"]
    _affected_by_seed = False

    # def __getstate__(self):
    #     # Copy the object's state from self.__dict__ which contains
    #     # all our instance attributes. Always use the dict.copy()
    #     # method to avoid modifying the original state.
    #     state = self.__dict__.copy()
    #     # Remove the unpicklable entries.
    #     del state['add_model_parameters']
    #     del state["parameters_combination_validator"]
    #     print(state)
    #     return state
    #
    # def __setstate__(self, state):
    #     # Restore instance attributes (i.e., filename and lineno).
    #     self.__dict__.update(state)

    def __init__(self, args):

        assert len(args.conv_layers) == len(args.heads)

        super(GAT_Model, self).__init__()

        self.features = args.features
        self.num_features = len(self.features)
        self.conv_layers = args.conv_layers
        self.heads = args.heads
        self.fc_layers = args.fc_layers
        self.concat = args.concat
        self.negative_slope = args.negative_slope
        self.dropout = args.dropout
        self.bias = args.bias
        self.seed_train = args.seed_train

        # Call super

        self.convolutional_layers = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.fullyconnected_layers = torch.nn.ModuleList()

        # TODO support non constant concat values
        for i in range(len(self.conv_layers)):
            num_heads = self.heads[i - 1] if ((self.concat[i - 1] is True) and (i > 0)) else 1
            in_channels = self.conv_layers[i - 1] * num_heads if i > 0 else self.num_features
            self.convolutional_layers.append(
                GATConv(in_channels=in_channels,
                        out_channels=self.conv_layers[i],
                        heads=self.heads[i],
                        concat=self.concat[i],
                        negative_slope=self.negative_slope[i],
                        dropout=self.dropout[i],
                        bias=self.bias[i])
            )

            num_out_heads = self.heads[i] if self.concat[i] is True else 1
            self.linear_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.conv_layers[i] * num_out_heads)
            )

        # Regressor

        # If last layer output is not a regressor, append a layer
        if self.fc_layers[-1] != 1:
            self.fc_layers.append(1)

        for i in range(len(self.fc_layers)):
            num_heads = self.heads[-1] if ((self.concat[-1] is True) and (i == 0)) else 1
            in_channels = self.fc_layers[i - 1] if i > 0 else self.conv_layers[-1] * num_heads
            self.fullyconnected_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.fc_layers[i])
            )

    def forward(self, x, edge_index):

        for i in range(len(self.convolutional_layers)):
            x = F.elu(self.convolutional_layers[i](x, edge_index) + self.linear_layers[i](x))

        x = x.view(x.size(0), -1)
        for i in range(len(self.fullyconnected_layers)):
            # TODO ELU?
            x = F.elu(self.fullyconnected_layers[i](x))

        x = x.view(x.size(0))
        x = torch.sigmoid(x)
        # print(x.size())
        return x

    @staticmethod
    def add_model_parameters(parser, grid=False):

        action = "append" if grid else None
        wrapper = (lambda x: [x] if grid else x)

        # def wrapper(x):
        #     if grid:
        #         return [x]
        #     else:
        #         return x

        parser.add_argument(
            "-CL",
            "--conv_layers",
            type=int,
            nargs="*",
            # default=wrapper([10]),
            required=True,
            action=action,
            help="",
        )
        parser.add_argument(
            "-H",
            "--heads",
            type=int,
            nargs="*",
            # default=wrapper([1]),
            required=True,
            action=action,
            help="",
        )
        parser.add_argument(
            "-FCL",
            "--fc_layers",
            type=int,
            nargs="*",
            # default=wrapper([100]),
            action=action,
            required=True,
            help="",
        )
        parser.add_argument(
            "-C",
            "--concat",
            type=bool,
            nargs="*",
            default=wrapper(DefaultDict(True)),
            action=action,
            help="",
        )
        parser.add_argument(
            "-NS",
            "--negative_slope",
            type=float,
            nargs="*",
            default=wrapper(DefaultDict(0.2)),
            action=action,
            help="",
        )
        parser.add_argument(
            "-d",
            "--dropout",
            type=float,
            nargs="*",
            default=wrapper(DefaultDict(0.3)),
            action=action,
            help="Dropout rate for the model, between 0.0 and 1.0",
        )
        parser.add_argument(
            "-B",
            "--bias",
            type=bool,
            nargs="*",
            default=wrapper(DefaultDict(True)),
            action=action,
            help="",
        )

    def add_run_parameters(self, run: dict):
        for parameter in self._model_parameters:
            if parameter != "fc_layers":
                num_layers = len(self.conv_layers)
            else:
                num_layers = len(self.fc_layers)

            run[parameter] = ','.join(str(vars(self)[parameter][i]) for i in range(num_layers)) + ","

        # run["seed"] = self.seed_test

    def model_name(self):
        name = []
        for parameter in self._model_parameters:
            if parameter != "fc_layers":
                num_layers = len(self.conv_layers)
            else:
                num_layers = len(self.fc_layers)

            name.append("{}{}".format(''.join(x[0].upper() for x in parameter.split("_")),
                                      '_'.join(str(vars(self)[parameter][i]) for i in range(num_layers))
                                      )
                        )
        name.append("S{}".format(self.seed_train))

        return '_'.join(name)

    @staticmethod
    def parameters_combination_validator(params):
        if len(params["conv_layers"]) != len(params["heads"]):
            return False

        return dotdict(params)
