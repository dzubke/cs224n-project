
from src.data_utils.load_data import get_all_instances
import random
import jsonlines
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Path to natural-instructions/"
)
parser.add_argument(
    "--files",
    type=str,
    help="Comma separated files for task lists"
)

parser.add_argument(
    "--sample_num",
    type=int,
    default=3250
)

def write_jsonl(output_path, instances):
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(instances)

if __name__=="__main__":
    args = parser.parse_args()
    task_map = dict()
    for file in args.files.split(","):
        with open(file) as f:
            name = file.split("/")[-1].split(".")[0]
            tasks = json.load(f)
            task_map[name] = tasks
    for name, tasks in task_map.items():
        instances = get_all_instances(args.dataset_dir, tasks, args.sample_num)
        write_jsonl(f"src/data/sampled/{name}_sampled_{len(instances)}.jsonl", instances)

