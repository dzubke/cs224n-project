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
    all_examples = []
    header = ""
    with open(args.csv, encoding='utf-8') as f:
        reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
        header = next(reader)
        total_rows = 0
        for row in reader:
            all_examples.append(row)
            total_rows += 1
    print("total_rows:", total_rows)
    total_examples = len(all_examples)
    print("Num examples:", total_examples)
    if args.sample_num:
        sample_num = int(args.sample_num)
    else:
        sample_num = int(args.sample_rate * total_examples)
    samples = random.sample(all_examples, sample_num)
    print("Num samples:", len(samples))
    with open(args.csv[:-4]+"_sampled_{}.csv".format(sample_num), "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sample in samples:
            writer.writerow(sample)



if __name__=="__main__":
    main(parser.parse_args())