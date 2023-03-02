import json
import csv
import argparse
import pdb
import random
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv",
    type=str,
    help="Path to JSON file with the training tasks."
)

parser.add_argument(
    "--sample_rate",
    type=float,
    help="Percentage of dataset to sample."
)

parser.add_argument(
    "--sample_num",
    type=int,
    help="Number of dataset to sample. Overrides sample_rate."
)

def main(args):
    train_map = defaultdict(list)
    header = ""
    with open(args.csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            task_name,id,definition,inputs,targets = row
            train_map[id] = row
    total_examples = len(train_map.keys())
    print("Num training examples:", total_examples)
    if args.sample_num:
        sample_num = int(args.sample_num)
    else:
        sample_num = int(args.sample_rate * total_examples)
    samples = random.sample(train_map.keys(), sample_num)
    print("Num samples:", len(samples))
    with open(args.csv[:-4]+"_sampled.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sample in samples:
            writer.writerow(train_map[sample])



if __name__=="__main__":
    main(parser.parse_args())