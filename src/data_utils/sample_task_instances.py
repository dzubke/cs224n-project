import pdb
import json
from datasets import load_dataset, DatasetDict
from src.data_utils.load_data import read_json_file
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import jsonlines
import random

instances_map_path = "/Users/hongjeon/scpd/224n/cs224n-project/src/data_utils/split_instance_count.json"
dataset_path = "/Users/hongjeon/scpd/224n/cs224n-project/natural-instructions"
similar_categories = ['Question Answering', 'Question Understanding', 'Question Generation', 'Misc.', 'Text Categorization', 'Fill in The Blank', 'Commonsense Classification', 'Information Extraction', 'Text Completion', 'Text Matching']
different_categories = ['Toxic Language Detection', 'Story Composition', 'Text to Code', 'Pos Tagging', 'Dialogue Generation', 'Program Execution', 'Sentiment Analysis', 'Wrong Candidate Generation', 'Code to Text', 'Linguistic Probing']
# Number of task instances to sample per each category.
sample_num = 40000

def write_jsonl(output_path, mapping):
    # Flatten mapping
    flattened = []
    for k,v in mapping.items():
        flattened += v
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(flattened)

if __name__=="__main__":
    category_to_instance = json.load(open(instances_map_path, encoding="utf-8"))
    similar_categories_map = defaultdict(list)
    different_categories_map = defaultdict(list)
    for category, val in tqdm(category_to_instance['train']['category_to_instance'].items()):
        if category not in similar_categories and category not in different_categories:
            continue
        elif category in similar_categories:
            mapping = similar_categories_map
        else:
            mapping = different_categories_map
        for task, count in val.items():
            if task == "total":
                continue
            try:
                filepath = Path(dataset_path) / "tasks" / f"{task}.json"
                ds = read_json_file(filepath)[task]
            except:
                print(f"ERROR. {filepath} does not exist.")
                continue
            # print(f"{category} | {task} | {len(ds['Instances'])}")
            for instance in ds['Instances']:
                for output in instance['output']:
                    mapping[category].append({
                        "task_name": task,
                        "id": instance['id'],
                        # Why is definition a list?
                        "definition": ds['Definition'][0],
                        "inputs": instance['input'],
                        "targets": output
                    })
    total_tasks = 0
    for key,value in different_categories_map.items():
        tasks = set()
        for instance in value:
            tasks.add(instance['task_name'])            
        total_tasks += len(tasks)
    print("different", total_tasks)

    total_tasks = 0
    for key,value in similar_categories_map.items():
        tasks = set()
        for instance in value:
            tasks.add(instance['task_name'])            
        total_tasks += len(tasks)
    print("similar", total_tasks)
    
    similar_categories_sampled = {}
    different_categories_sampled = {}
    print("Similar categories")
    for category, instances in similar_categories_map.items():
        print(f"{category} | {len(instances)}")
        samples = random.sample(instances, sample_num)
        similar_categories_sampled[category] = samples
    print("Different categories")
    for category, instances in different_categories_map.items():
        print(f"{category} | {len(instances)}")
        samples = random.sample(instances, sample_num)
        different_categories_sampled[category] = samples
    
    write_jsonl(f"src/data/sampled/similar_categories_sampled_{sample_num}.jsonl", similar_categories_sampled)
    write_jsonl(f"src/data/sampled/different_categories_sampled_{sample_num}.jsonl", different_categories_sampled)