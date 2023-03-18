import json
import csv
import argparse
import pdb
import random
import os
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tasks",
    type=str,
    help="Path to JSON file with the training tasks."
)

parser.add_argument(
    "--output_dir",
    type=str,
)

parser.add_argument(
    "--sample_num",
    type=int,
    help="Number of dataset to sample. Overrides sample_rate."
)

if __name__=="__main__":
    args = parser.parse_args()
    with open(args.tasks) as f:
        tasks = json.load(f)
    sampled = random.sample(tasks, args.sample_num)
    with open(os.path.join(args.output_dir, f"tasks_random_{args.sample_num}.json"), 'w') as out:
        json.dump(sampled, out, indent=4)