
from src.data_utils.load_data import get_first_instances
import os
import jsonlines

dataset_path = "/Users/hongjeon/scpd/224n/cs224n-project/natural-instructions"
test_tasks="splits/default/test_tasks.txt"

def write_jsonl(output_path, instances):
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(instances)

if __name__=="__main__":
    tasks_path = os.path.join(dataset_path, test_tasks)
    tasks=[]
    with open(tasks_path) as f:
        for line in f:
            tasks.append(line.strip())
    print(f"Have {len(tasks)} tasks")
    instances = get_first_instances(dataset_path, tasks)
    print(f"got {len(instances)} instances")
    write_jsonl(f"src/data/sampled/test_{len(instances)}.jsonl", instances)

