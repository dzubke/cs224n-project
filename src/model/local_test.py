from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
import torch
import time
import pdb

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def encode_input_def_pos1(definition, input):
    return "Definition: {} | Input: {}".format(definition, input)

def tokenize_input_and_target(examples):
    input_strings = [encode_input_def_pos1(x,y) for x,y in zip(examples['definition'],examples['inputs'])]
    targets = examples['targets']
    # Encode the inputs
    inputs = tokenizer(input_strings, return_tensors="pt", padding=True, truncation=True)
    # Encode the labels
    labels = tokenizer(targets, return_tensors="pt", padding=True, truncation=True).input_ids
    # Set loss to -100, which is ignored by CrossEntropyLoss.
    labels[labels == tokenizer.pad_token_id] = -100

    inputs['labels'] = labels
    return inputs

def load_tasks(tasks_path):
    tasks = set()
    with open(tasks_path, 'r') as f:
        for line in f.readlines():
            tasks.add(line.strip())
    return tasks

def main():
    start = time.time() 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = model.to(device)
    dataset = load_dataset("Muennighoff/natural-instructions")

    train_tasks = load_tasks("data/02_26_23/train_tasks.txt") 
    test_tasks = load_tasks("data/02_26_23/test_tasks.txt") 

    overfitting_dataset = DatasetDict(
        train=dataset['train'].filter(lambda example: example["task_name"] in train_tasks),
        val=dataset['validation'].filter(lambda example: example["task_name"] in test_tasks)
    )
    overfitting_dataset_tokenized = overfitting_dataset.map(tokenize_input_and_target, batched=True)
    
    arguments = TrainingArguments(
        output_dir="sample_hf_trainer",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=100,
        evaluation_strategy="epoch",  # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=1e-4,
        load_best_model_at_end=True,
        report_to="wandb",
        seed=224
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=overfitting_dataset_tokenized['train'],
        eval_dataset=overfitting_dataset_tokenized['val'],
        tokenizer=tokenizer,
    )

    trainer.train()

    diff = time.time() - start
    print("Total time: {} seconds or {} minutes".format(diff, diff/60.0))

if __name__=="__main__":
    main()