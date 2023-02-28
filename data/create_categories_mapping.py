"""Autogenerate train/test splits by running the script.

python create_train_test_splits.py --output_dir=<name>

Default flags only work if you run the command from this folder.

"""
import json
import argparse
from os import listdir, path
from os.path import isfile, join
from collections import defaultdict
import os
import glob
import pdb

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tasks_path",
    type=str,
    default="tasks/",
    help="the directory path of all the task json files in NaturalInstructions."
)

parser.add_argument(
    "--output_path",
    type=str,
    default="tasks.txt",
    help="Output path of where to write the tasks file."
)

parser.add_argument(
    "--track",
    choices=["default", "xlingual"],
    required=False,
    default="default",
    help="`default` will generate the splits for the English-only track, `xlingual` will generate the splits for the cross-lingual track."
)

def create_category_mapping(tasks_path, output_path, track):
    category_mapping = defaultdict(list)
    for file in glob.glob(os.path.join(tasks_path, "task*.json")):
        task = os.path.basename(file)[:-5]
        print(task)
        try:
            task_info = json.load(open(file, encoding="utf-8"))
        except:
            print("Error reading task:{}".format(task))
            continue
        languages = set(task_info["Input_language"] + task_info["Output_language"])
        if (len(languages)==1 and "English" not in languages) or len(languages) > 1:
            # Skip cross lingual tasks if we are on the default track.
            if track == "default":
                print("Skipping non-english task:{}".format(task))
                continue
        categories = task_info['Categories']
        for category in categories:
            category = category.strip()
            category_mapping[category].append(task)
    with open(output_path, 'w') as o:
        dump = json.dumps(category_mapping, indent = 4, ensure_ascii=False)
        o.write(dump)

if __name__ == "__main__":
    args = parser.parse_args()
    create_category_mapping(args.tasks_path, args.output_path, args.track)