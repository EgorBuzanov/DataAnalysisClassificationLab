import pickle
from ruamel.yaml import YAML
from typing import Any
from pandas import DataFrame


def get_params() -> dict:
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    print(type(params))
    return params


def save_as_pickle(obj: Any, path: str) -> None:
    if isinstance(obj, DataFrame):
        obj.to_pickle(path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def load_pickle(path: str) -> None:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model