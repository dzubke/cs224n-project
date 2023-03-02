import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.model.dataset import TaskDataset
import argparse
import pdb
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="The directory containing the csv files to load train and test dataset."
)

parser.add_argument(
    "--train_file",
    type=str,
)

parser.add_argument(
    "--test_file",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()
    rouge = evaluate.load("rouge")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("/Users/hongjeon/scpd/224n/cs224n-project/checkpoints/checkpoint-10716")
    train_ds, test_ds = TaskDataset(args.dataset_dir, args.train_file, args.test_file).get_dataset()
    print("columns:", test_ds.features)
    for example in test_ds:
        print("--------")
        ids = example['input_ids']
        inputs = torch.IntTensor(ids)
        inputs = inputs.unsqueeze(dim=0)
        inputs_decoded = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        print("inputs: ", inputs_decoded)
        outputs = model.generate(input_ids=inputs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("outputs:", outputs)
        print("reference:", [example['targets']])
        results = rouge.compute(predictions=outputs, references=[example['targets']])
        print(results)