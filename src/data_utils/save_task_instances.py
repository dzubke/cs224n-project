
from src.data_utils.load_data import get_all_instances
import random
import jsonlines
import json

dataset_path = "/Users/hongjeon/scpd/224n/cs224n-project/natural-instructions"
sample_num = 3250

def write_jsonl(output_path, instances):
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(instances)

if __name__=="__main__":
    with open("/Users/hongjeon/scpd/224n/cs224n-project/src/data/all_task_groupings.json") as f:
        task_map = json.load(f)
    for name, tasks in task_map.items():
        instances = get_all_instances(dataset_path, tasks, sample_num)
        write_jsonl(f"src/data/sampled/{name}_sampled_{len(instances)}.jsonl", instances)

