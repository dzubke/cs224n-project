# standard libs
import json
from pathlib import Path


def load_task_data(data_path):
    data_path = Path(data_path)
    if data_path.is_dir():
        data = {}
        for json_path in data_path.rglob("*.json"):
            raise NotImplementedError
    else:
        if data_path.suffix == ".json":
            with open(data_path, "r") as fid:
                data = {"outer_key": json.load(fid)}

    return data
