from collections import defaultdict
import json

from src.data_utils.load_data import load_task_data
from structures import SupNatKeys, InstanceKeys


def compute_statistics():
    data_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/data/natural-instructions/tasks"
    data = load_task_data(data_path)

    category_to_instance_count = defaultdict(int)
    for contents in data.values():
        count = 0
        instances = contents[SupNatKeys.INSTANCES.value]
        for instance in instances:
            count += len(instance[InstanceKeys.OUTPUT.value])

        categories = contents[SupNatKeys.CATEGORIES.value]
        for cat in categories:
            category_to_instance_count[cat] += count

    with open("category_instance_count.json", "w") as fid:
        json.dump(category_to_instance_count, fid)


if __name__ == "__main__":
    compute_statistics()
