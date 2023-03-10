import argparse
import pdb
import json
from typing import Dict, List, Union
import pickle

from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5ForConditionalGeneration
import torch

from src.model.dataset import TaskDataset


def write_predictions(run_name, model_path, verbose=False):

    dataset = load_dataset("jayelm/natural-instructions")
    test_data = dataset['test'].filter(lambda e: e['eval'] == True)

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    task_dataset = TaskDataset(dataset_dir="", train_file="", test_file="")
    tokenized_dataset = test_data.map(
        task_dataset._tokenize_input_and_target, batched=True, batch_size=4, num_proc=4
    )
    references = []
    predictions = []
    total = len(tokenized_dataset)
    for i in range(len(tokenized_dataset)):
        example = tokenized_dataset[i]
        ids = example["input_ids"]
        inputs = torch.IntTensor(ids)
        inputs = inputs.unsqueeze(dim=0)
        outputs = model.generate(input_ids=inputs)
        outputs = task_dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references.append(
            {
                "id": example["id"],
                "references": [example["targets"]],
                "task_id": example["task_name"],
                "track": "default",
            }
        )
        assert len(outputs) == 1
        predictions.append(
            {
                "id": example["id"],
                "prediction": outputs[0],
                "task_id": example["task_name"],
                "track": "default",
            }
        )

        if verbose:
            # inputs_decoded = task_dataset.tokenizer.batch_decode(inputs, skip_special_tokens=True)
            # print("inputs: ", inputs_decoded)
            print(f"{i} of {total}")
            print("outputs:", outputs)
            print("reference:", [example["targets"]])
            print()

    write_jsonl_file(references, f"val_references_{run_name}.jsonl", target_key="references")
    write_jsonl_file(predictions, f"val_predictions_{run_name}.jsonl", target_key="prediction")


def write_jsonl_file(dataset: Union[Dataset, List[Dict[str, str]]], write_path, target_key):

    with open(write_path, "w") as fid:
        for i in range(len(dataset)):
            datum = dataset[i]
            out_json = {
                "id": datum["id"],
                target_key: datum[target_key],
                "task_id": datum["task_id"],
                "track": "default",
            }
            json.dump(out_json, fid)
            fid.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    write_predictions(verbose=args.verbose, run_name=args.run_name, model_path=args.model_path)
