# standard libs
import json
from pathlib import Path
from typing import Dict, List

from src.data_utils.constants import (
    ExampleKeys, InstanceKeys, SupNatKeys,
    DATASET, FILE_CONTENTS, TASK_INSTANCE, 
)

JOIN_STR = "__"

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
    for task_filename, file_contents in raw_dataset.items():
        extracted_contents = extract_contents(task_filename, file_contents)
        output.update(extracted_contents)
    return output

def extract_contents(task_filename: str, file_contents: FILE_CONTENTS) -> Dict[str, str]:
    extracted_contents = {}

    defintion = file_contents[SupNatKeys.DEFINTION.value]
    pos_examples = file_contents[SupNatKeys.POSITIVE_EXAMPLES.value]
    neg_examples = file_contents[SupNatKeys.NEGATIVE_EXAMPLES.value]

    extracted_pos_examples: str = extract_examples(pos_examples, example_type='positive')
    extracted_neg_examples: str = extract_examples(neg_examples, example_type='negative')

    output_str = defintion
    output_str += extracted_pos_examples
    output_str += extracted_neg_examples

    extracted_contents = {task_filename: output_str}

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

    return extracted_contents

def extract_examples(examples: List[Dict[str, str]], example_type) -> str:
    assert example_type in ['positive', 'negative']
    for id, example in enumerate(examples):
        output_str = ''
        output_str += f"{example_type} example {id} {ExampleKeys.INPUT.value}: {example[ExampleKeys.INPUT.value]}"
        output_str += f"{example_type} example {id} {ExampleKeys.OUTPUT.value}: {example[ExampleKeys.OUTPUT.value]}"
        output_str += f"{example_type} example {id} {ExampleKeys.EXPLANATION.value}: {example[ExampleKeys.EXPLANATION.value]}"
    return output_str

def extract_task_outputs(outputs) -> str:
    return outputs[0]