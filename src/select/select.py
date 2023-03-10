import json


def main(count_path):
    count_threshold = 40_000
    selected_categories = select_categories_by_instance_threshold(count_path, count_threshold)


def select_categories_by_instance_threshold(count_path, count_threshold=40_000):
    with open(count_path) as fid:
        count = json.load(fid)

    selected_categories = []
    for cat, count in count["train"]["category_to_instance"].items():
        if count > count_threshold:
            selected_categories.append(cat)

    return selected_categories


if __name__ == "__main__":
    count_path = "/Users/dustin/Documents/School/1_stanford/classes/cs224n/project/cs224n-project/src/data_utils/split_instance_count_simple.json"
    main(count_path)
