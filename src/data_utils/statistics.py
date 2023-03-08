from collections import defaultdict
import json
import os

from src.data_utils.load_data import load_task_data
from structures import SupNatKeys, InstanceKeys


def compute_statistics():
    data_dir_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/data/natural-instructions/"

    data = load_task_data(os.path.join(data_dir_path, "tasks"))
    train_task_set, test_task_set = load_train_test_task_sets(
        os.path.join(data_dir_path, "splits", "default")
    )

    count_dict = {
        "train": {"category_to_instance": defaultdict(int), "task_to_instance": defaultdict(int)},
        "test": {"category_to_instance": defaultdict(int), "task_to_instance": defaultdict(int)},
    }
    excluded_tasks = []
    for task_name, contents in data.items():

        if task_name in train_task_set:
            split_key = "train"
        elif task_name in test_task_set:
            split_key = "test"
        else:
            excluded_tasks.append(task_name)
            continue

        instance_count = 0
        instances = contents[SupNatKeys.INSTANCES.value]
        for instance in instances:
            instance_count += len(instance[InstanceKeys.OUTPUT.value])

        categories = contents[SupNatKeys.CATEGORIES.value]
        for cat in categories:
            count_dict[split_key]["category_to_instance"][cat] += instance_count
        count_dict[split_key]["task_to_instance"][task_name] = instance_count

    with open("split_instance_count.json", "w") as fid:
        json.dump(count_dict, fid)

    with open("excluded_tasks.json", "w") as fid:
        json.dump(excluded_tasks, fid)


def load_train_test_task_sets(data_path):
    task_sets = []
    for task_filename in ["train_tasks.txt", "test_tasks.txt"]:
        with open(os.path.join(data_path, task_filename)) as fid:
            task_set = set([line.strip() for line in fid.readlines()])
            task_sets.append(task_set)
    return task_sets[0], task_sets[1]


def split_key_switch(task_name, train_task_set, test_task_set):

    return split_key


if __name__ == "__main__":
    compute_statistics()
