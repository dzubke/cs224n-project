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
    "--category_mapping_path",
    type=str,
    default="categories_mapping.json",
    help="the json file containing category to tasks mapping."
)

parser.add_argument(
    "--train_categories_path",
    type=str,
    default="train_categories.txt",
    help="The text file containing training categories."
)

parser.add_argument(
    "--test_categories_path",
    type=str,
    default="test_categories.txt",
    help="The text file containing testing categories"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="temp",
    help="Output directory of where to output train and test text files."
)

def create_train_test_split(category_mapping_path, output_dir, train_categories_path, test_categories_path):
    category_mapping = json.load(open(category_mapping_path, encoding="utf-8"))
    train_categories = set()
    test_categories = set()
    with open(train_categories_path, "r") as f, open(output_dir+"/train_tasks.txt", "w") as o:
        for line in f.readlines():
            category = line.strip()
            try:
                tasks = category_mapping[category]
                for task in tasks:
                    o.write(task + "\n")
                train_categories.add(category)
            except:
                print("No such category exists in mapping: {}".format(category))
    # with open(test_categories_path, "r") as f, open(output_dir+"/test_tasks.txt", "w") as o:
    #     for line in f.readlines():
    #         category = line.strip()
    #         try:
    #             tasks = category_mapping[category]
    #             for task in tasks:
    #                 o.write(task + "\n")
    #             test_categories.add(category)
    #         except:
    #             print("No such category exists in mapping: {}".format(category))
    with open(output_dir+"/categories.txt", "w") as o:
        o.write("Train categories:\n")
        for train_c in train_categories:
            o.write("\t" + train_c + "\n")
        # o.write("Test categories:\n")
        # for test_c in test_categories:
        #     o.write("\t" + test_c + "\n")


if __name__ == "__main__":
    args = parser.parse_args()
    create_train_test_split(args.category_mapping_path, args.output_dir, args.train_categories_path, args.test_categories_path)