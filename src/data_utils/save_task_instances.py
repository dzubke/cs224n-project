
from src.data_utils.load_data import get_all_instances
import random
import jsonlines
import json


files="/Users/hongjeon/scpd/224n/cs224n-project/src/data/tasks_random_50.json,/Users/hongjeon/scpd/224n/cs224n-project/src/data/tasks_random_100.json,/Users/hongjeon/scpd/224n/cs224n-project/src/data/tasks_random_200.json"
dataset_path = "/Users/hongjeon/scpd/224n/cs224n-project/natural-instructions"
sample_num = 3250

def write_jsonl(output_path, instances):
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(instances)

if __name__=="__main__":
    task_map = dict()
    for file in files.split(","):
        with open(file) as f:
            name = file.split("/")[-1].split(".")[0]
            tasks = json.load(f)
            task_map[name] = tasks
    for name, tasks in task_map.items():
        instances = get_all_instances(dataset_path, tasks, sample_num)
        write_jsonl(f"src/data/sampled/{name}_sampled_{len(instances)}.jsonl", instances)

