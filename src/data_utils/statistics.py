from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

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

    count_dict = sort_by_count(count_dict)
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


def sort_by_count(count: dict) -> dict:
    for split_name in ["train", "test"]:
        for count_name in ["category_to_instance", "task_to_instance"]:
            count[split_name][count_name] = dict(
                sorted(count[split_name][count_name].items(), key=lambda x: x[1], reverse=True)
            )
    return count


def plot_historgrams(split_name, count_name, bins=50, ticks=list(range(0, 500000, 20000)):
    with open("split_instance_count.json", "r") as fid:
        count = json.load(fid)

    df = pd.DataFrame(count[split_name][count_name].items(), columns=[count_name, "count"])
    df["count"].plot.hist(grid=True, bins=bins, rwidth=0.9, color="#607c8e")
    plt.xticks(ticks=))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    plt.xlabel("Task instance count")
    plt.ylabel(count_name)
    plt.show()


if __name__ == "__main__":
    compute_statistics()
