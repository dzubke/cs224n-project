# standard libs
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

from src.data_utils.structures import (
    ExampleKeys,
    InstanceKeys,
    ProjectKeys,
    SupNatKeys,
    DATASET,
    FILE_CONTENTS,
    TASK_INSTANCE,
)

JOIN_STR = "__"


def load_tasks_set(tasks_path):
    """Reads the tasks from the path and returns a set."""
    tasks = set()
    with open(tasks_path, "r") as f:
        for line in f.readlines():
            tasks.add(line.strip())
    return tasks


def load_task_data(data_path):
    data_path = Path(data_path)
    file_paths = data_path.rglob("*.json") if data_path.is_dir() else [data_path]
    dataset: DATASET = {}
    for file_path in file_paths:
        dataset.update(read_json_file(file_path))

    return dataset


def read_json_file(file_path) -> DATASET:
    with open(file_path, "r") as fid:
        task_filename = file_path.stem
        raw_instances: TASK_INSTANCE = json.load(fid)
        return {task_filename: raw_instances}


def load_semantic_sim_data(data_path: str) -> dict:
    raw_dataset: DATASET = load_task_data(data_path)
    output = {}
    task_to_category = {}
    task_to_source = {}
    for task_filename, file_contents in raw_dataset.items():
        extracted_contents, task_to_cat, task_to_src = extract_contents(
            task_filename, file_contents
        )
        output.update(extracted_contents)
        task_to_category.update(task_to_cat)
        task_to_source.update(task_to_src)
    return output, task_to_category, task_to_source


def extract_contents(
    task_filename: str, file_contents: FILE_CONTENTS
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    extracted_contents = {}

    categories = file_contents[SupNatKeys.CATEGORIES.value]
    definition = file_contents[SupNatKeys.DEFINTION.value]
    pos_examples = file_contents[SupNatKeys.POSITIVE_EXAMPLES.value]
    neg_examples = file_contents[SupNatKeys.NEGATIVE_EXAMPLES.value]

    extracted_pos_examples: str = extract_examples(pos_examples, example_type="positive")
    extracted_neg_examples: str = extract_examples(neg_examples, example_type="negative")

    output_str = f"Categories: {' '.join(categories)} | "
    output_str += f"Definition: {definition} | "
    output_str += extracted_pos_examples
    output_str += extracted_neg_examples

    extracted_contents = {task_filename: output_str}
    task_to_categories = {task_filename: file_contents[SupNatKeys.CATEGORIES.value]}
    task_to_source = {task_filename: file_contents[SupNatKeys.SOURCE.value]}

    ## not considering task instances yet
    # for task_instance in file_contents[SupNatKeys.INSTANCES.value]:
    #     task_id: str = task_instance[InstanceKeys.ID.value]
    #     task_input: str = task_instance[InstanceKeys.INPUT.value]
    #     task_outputs: List[str] = task_instance[InstanceKeys.OUTPUT.value]
    #     extracted_outputs: str = extract_task_outputs(task_outputs)

    #     output_str = extracted_pos_examples
    #     output_str += extracted_neg_examples
    #     output_str += f"task input: {task_input}"
    #     output_str += f"task outputs: {extracted_outputs}"

    #     full_id = task_filename + JOIN_STR + task_id
    #     extracted_contents.update({full_id: output_str})

    return extracted_contents, task_to_categories, task_to_source


def extract_examples(examples: List[Dict[str, str]], example_type) -> str:
    assert example_type in ["positive", "negative"]
    for id, example in enumerate(examples):
        output_str = ""
        output_str += f"{example_type} example {id} {ExampleKeys.INPUT.value}: {example[ExampleKeys.INPUT.value]}, "
        output_str += f"{example_type} example {id} {ExampleKeys.OUTPUT.value}: {example[ExampleKeys.OUTPUT.value]}, "
        output_str += f"{example_type} example {id} {ExampleKeys.EXPLANATION.value}: {example[ExampleKeys.EXPLANATION.value]}, "
    return output_str


def extract_task_outputs(outputs) -> str:
    return outputs[0]

def get_all_instances(dataset_path :str, tasks : List[str], sample_num):
    """Given a list of tasks, return a list of task instance objects used for training."""
    instances = []
    for task in tasks:
        if task == "total":
            continue
        try:
            filepath = Path(dataset_path) / "tasks" / f"{task}.json"
            ds = read_json_file(filepath)[task]
        except:
            print(f"ERROR. {filepath} does not exist.")
            continue
        temp_instances = []
        for instance in ds['Instances']:
            for output in instance['output']:
                temp_instances.append({
                    "task_name": task,
                    "id": instance['id'],
                    # Why is definition a list?
                    "definition": ds['Definition'][0],
                    "inputs": instance['input'],
                    "targets": output
                })
        samples = random.sample(temp_instances, sample_num)
        instances += samples
    return instances

    
