import json
import csv
import argparse
import pdb
import random
from collections import defaultdict
from src.model.dataset import TaskDataset
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Path to JSON file with the training tasks."
)

def main(args):
    tasks = TaskDataset(args.dataset_dir, "train.csv", "test.csv")
    tasks.output_csv()

if __name__=="__main__":
    main(parser.parse_args())