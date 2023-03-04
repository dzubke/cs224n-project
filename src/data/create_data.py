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

def create_dict(task_name, task, instance):
    definition = task['Definition'][0]
    input = instance['input']
    # TODO: come up with a smarter way to take outputs. Take the first output for now.
    output = instance['output'][0]
    return {'task_name': task_name, 'id': instance['id'], 'definition': definition, 'inputs': input, 'targets': output}

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
        # Sample instances per task.
        for task, task_name in task_data:
            sampled = random.sample(task['Instances'], min(sample_num,len(task['Instances'])))
            for sample in sampled:
                sampled_instances.append(create_dict(task_name, task,sample))
        print(len(sampled_instances))
        task_filename = args.tasks.split("/")[-1][:-4]
        with open(args.experiment_dir+f"/sampled_{task_filename}_{args.sample_num}.json", "w", encoding='utf-8', newline='') as f: 
            json.dump({'data': sampled_instances}, f)

        
        



    

if __name__=="__main__":
    main(parser.parse_args())