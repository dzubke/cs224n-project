from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.cluster import pair_confusion_matrix

from src.data_utils.load_data import load_task_data
from structures import SupNatKeys, InstanceKeys


def compute_statistics():
    data_dir_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/data/natural-instructions/"

    data = load_task_data(os.path.join(data_dir_path, "tasks"))
    train_task_set, test_task_set = load_train_test_task_sets(
        os.path.join(data_dir_path, "splits", "default")
    )

    count_dict = {
        "train": {
            "category_to_instance": defaultdict(lambda: defaultdict(int)),
            "task_to_instance": defaultdict(int),
        },
        "test": {
            "category_to_instance": defaultdict(lambda: defaultdict(int)),
            "task_to_instance": defaultdict(int),
        },
    }
    excluded_tasks = []
    excluded_count = 0
    for task_name, contents in data.items():

        if task_name in train_task_set:
            split_key = "train"
        elif task_name in test_task_set:
            split_key = "test"
        else:
            excluded_tasks.append(task_name)
            excluded_count
            continue

        instance_count = 0
        instances = contents[SupNatKeys.INSTANCES.value]
        for instance in instances:
            instance_count += len(instance[InstanceKeys.OUTPUT.value])

        categories = contents[SupNatKeys.CATEGORIES.value]
        for cat in categories:
            count_dict[split_key]["category_to_instance"][cat]["total"] += instance_count
            count_dict[split_key]["category_to_instance"][cat][task_name] = instance_count

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

        # sort the tasks within a category
        for cat_name in count[split_name]["category_to_instance"]:
            count[split_name]["category_to_instance"][cat_name] = dict(
                sorted(
                    count[split_name]["category_to_instance"][cat_name].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        # sort the categories by the total task instances
        count[split_name]["category_to_instance"] = dict(
            sorted(
                count[split_name]["category_to_instance"].items(),
                key=lambda x: x[1]["total"],
                reverse=True,
            )
        )

        # sort task instances
        count[split_name]["task_to_instance"] = dict(
            sorted(count[split_name]["task_to_instance"].items(), key=lambda x: x[1], reverse=True)
        )
    return count


def plot_historgrams(split_name, count_name, bins=50, ticks=list(range(0, 500000, 20000))):
    with open("split_instance_count_simple.json", "r") as fid:
        count = json.load(fid)

    df = pd.DataFrame(count[split_name][count_name].items(), columns=[count_name, "count"])
    df["count"].plot.hist(grid=True, bins=bins, rwidth=0.9, color="#607c8e")
    plt.xticks(ticks=ticks)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    plt.xlabel("Task instance count")
    plt.ylabel(count_name)
    plt.show()


def plot_historgrams_report(split_name, count_name, bins=50, ticks=list(range(0, 100000, 10000))):
    with open("split_instance_count_simple.json", "r") as fid:
        count = json.load(fid)

    plt.figure(figsize=(5, 3))
    df = pd.DataFrame(count[split_name][count_name].items(), columns=[count_name, "count"])
    df["count"].plot.hist(grid=True, bins=bins, rwidth=0.9, color="#607c8e")
    plt.xticks(ticks=ticks)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    plt.xlabel("Instance Count per Task")
    plt.ylabel("Number of Tasks")
    plt.title("Histogram of Tasks by Instance Count")
    plt.tight_layout()
    plt.savefig(f"task-train-histogram.png")
    plt.show()


def plot_historgrams_report(split_name, count_name, bins=50, ticks=list(range(0, 500000, 50000))):
    with open("split_instance_count_simple.json", "r") as fid:
        count = json.load(fid)

    plt.figure(figsize=(5, 3))
    df = pd.DataFrame(count[split_name][count_name].items(), columns=[count_name, "count"])
    df["count"].plot.hist(grid=True, bins=bins, rwidth=0.9, color="#607c8e")
    plt.xticks(ticks=ticks)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    plt.xlabel("Instance Count per Task")
    plt.ylabel("Number of Categories")
    plt.title("Histogram of Categories by Instance Count")
    plt.tight_layout()
    plt.savefig(f"category-train-histogram.png")
    plt.show()


def calc_hist_from_input(embed_names, embed_to_val_path, write_path):
    with open(embed_to_val_path, "r") as fid:
        embed_to_val = json.load(fid)

    val_count = defaultdict(int)
    for name in embed_names:
        val = embed_to_val[name]
        if isinstance(val, list):
            assert len(val) == 1
            val = val[0]
        val_count[val] += 1

    val_count = dict(sorted(val_count.items(), key=lambda x: x[1], reverse=True))
    with open(write_path, "w") as fid:
        json.dump(val_count, fid)


def calc_cluster_pair_confusion_matrix(cluster_path, task_to_category_path):

    with open(task_to_category_path, "r") as fid:
        task_to_cat = json.load(fid)

    with open(cluster_path, "r") as fid:
        cluster_to_task = json.load(fid)

    assert all([len(x) == 1 for x in task_to_cat.values()])
    sorted_cats = sorted(set(x[0] for x in task_to_cat.values()))
    cat_to_int = {cat: i for i, cat in enumerate(sorted_cats)}
    cat_task_set = set(task_to_cat)
    clust_task_set = set([task for tasks in cluster_to_task.values() for task in tasks])
    assert cat_task_set == clust_task_set
    sorted_tasks = sorted(cat_task_set)

    task_to_cluster = {}
    for cluster, tasks in cluster_to_task.items():
        for task in tasks:
            task_to_cluster[task] = cluster

    category_cluster_list = [cat_to_int[task_to_cat[task][0]] for task in sorted_tasks]
    cluster_cluster_list = [task_to_cluster[task] for task in sorted_tasks]

    confusion_matrix = pair_confusion_matrix(category_cluster_list, cluster_cluster_list)
    print(confusion_matrix)


if __name__ == "__main__":
    compute_statistics()
