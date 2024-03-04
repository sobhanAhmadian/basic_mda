import json
import pickle
from typing import List, Union, Dict
from src.utils import prj_logger
import numpy as np

logger = prj_logger.getLogger(__name__)


def pickle_load(filename: str):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        print(f"Logging Info - Loaded: {filename}")
    except EOFError:
        print(f"Logging Error - Cannot load: {filename}")
        obj = None

    return obj


def json_dump(file_path, obj: Union[List, Dict]):
    try:
        with open(file_path, "w+") as f:
            json.dump(obj, f)
    except Exception as e:
        print(e)


def json_load(file_path) -> Union[List, Dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)


def pickle_dump(filename: str, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Logging Info - Saved: {filename}")


def dump_numpy_as_csv(filename: str, mat, sep=','):
    np.savetxt(filename, mat, delimiter=sep)
    logger.info(f"file with name {filename} saved!")
