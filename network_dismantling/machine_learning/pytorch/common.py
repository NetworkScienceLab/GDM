from graph_tool import unicode, Graph, load_graph_from_csv
from pathlib2 import Path


def load_graph(file, fmt="auto", ignore_vp=None, ignore_ep=None,
               ignore_gp=None, directed=True, **kwargs):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from graph_tool import load_graph

    if fmt == 'auto' and isinstance(file, (str, unicode)) and \
            Path(file).suffix[1:] in ["csv", "edgelist", "edge", "edges", "el"]:
        # g = load_edgelist(file, directed=directed)
        delimiter = "," if Path(file).suffix == ".csv" else " "
        g = load_graph_from_csv(file,
                                directed=directed,
                                eprop_types=kwargs.get("eprop_types", None),
                                eprop_names=kwargs.get("eprop_names", None),
                                string_vals=kwargs.get("string_vals", False),
                                hashed=kwargs.get("hashed", False),
                                skip_first=kwargs.get("skip_first", False),
                                ecols=kwargs.get("ecols", (0, 1)),
                                csv_options=kwargs.get("csv_options", {
                                    "delimiter": delimiter,
                                    "quotechar": '"'
                                }),
                                )
    else:
        g = load_graph(file, fmt=fmt, ignore_vp=ignore_vp, ignore_ep=ignore_ep, ignore_gp=ignore_gp, **kwargs)

    return g


# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(_callback=None, **kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()

    from itertools import product

    for instance in product(*vals):

        instance = dict(zip(keys, instance))
        if _callback:
            instance = _callback(instance)

            if not instance:
                continue

        yield instance


class DefaultDict(dict):
    def __init__(self, value):
        super().__init__()
        self.__default_value = value

    def __missing__(self, key):
        return self.__default_value


# Chi-squared ([o]bserved, [e]xpected)
def chi(o, e):
    if e == 0:
        return 0
    return (o - e) ** 2 / e
