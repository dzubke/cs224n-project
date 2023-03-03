import json
import csv
import argparse
import pdb
import random
from collections import defaultdict
from src.model.dataset import TaskDataset
from src.data_utils.load_data import load_task_data, read_json_file
import math
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Path to natural instructions repo."
)

parser.add_argument(
    "--experiment_dir",
    type=str,
    help="Path to write the output train and test files."
)

parser.add_argument(
    "--tasks",
    type=str,
    help="File path to tasks file to parse."
)

parser.add_argument(
    "--sample_num",
    type=int,
    help="Number of instance samples to include in total."
)

def dict_to_row(task_name, task, instance):
    # TODO: come up with a smarter way to take outputs. Take the first output for now.
    return [task_name, instance['id'], repr(str(task['Definition'][0])),repr(str(instance['input'])), repr(str(instance['output'][0]))]

def main(args):
    # Task id -> task
    task_data = []
    total_instances = 0
    total_tasks = 0
    with open(args.tasks, 'r') as f:
        for task in f:
            task = task.strip()
            total_tasks += 1
            try:
                filepath = Path(args.dataset_dir) / "tasks" / f"{task}.json"
                ds = read_json_file(filepath)[task]
            except:
                print(f"ERROR. {filepath} does not exist.")
                continue
            # TODO: support crosslingual. Only use english<->english tasks for now.
            if 'English' not in ds['Input_language'] or 'English' not in ds['Output_language']:
                print(f"Skipping non english task: {task}")
                continue
            task_data.append((ds, task))
            total_instances += len(ds['Instances'])
        sampled_instances = [] # each item is a csv row
        if args.sample_num > total_instances:
            print(f"{args.sample_num} is greater than the total number of instances: {total_instances}")
            return
        # Sample the same number of instances from each task.
        sample_num = math.floor(args.sample_num / total_tasks)
        print(f"Total tasks: {total_tasks}. Sampling {sample_num} instances from each task.")
        row_len = None
        for task, task_name in task_data:
            sampled = random.sample(task['Instances'], min(sample_num,len(task['Instances'])))
            for sample in sampled:
                row = dict_to_row(task_name, task,sample)
                sampled_instances.append(row)
                if row_len and row_len != len(row):
                    print("different length:", len(row))
                    pdb.set_trace()
                row_len = len(row)
        print(len(sampled_instances))
        task_filename = args.tasks.split("/")[-1][:-4]
        with open(args.experiment_dir+f"/sampled_{task_filename}_{args.sample_num}.csv", "w", encoding='utf-8', newline='') as f: 
            w = csv.writer(f)
            cols = ["task_name", "id", "definition", "inputs", "targets"]
            col_len = len(cols)
            row_len = len(sampled_instances)
            print(f"col_len: {col_len}, row_len: {row_len}")
            w.writerow(cols)
            w.writerows(sampled_instances)

        
        



    

if __name__=="__main__":
    main(parser.parse_args())