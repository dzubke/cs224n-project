
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch
import argparse
from src.model.dataset import TaskDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_dir",
    type=str,
    default="",
    help="The directory containing the csv files to load train and test dataset."
)

parser.add_argument(
    "--train_file",
    default="",
    type=str,
)

parser.add_argument(
    "--test_file",
    default="",
    type=str,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size used for training."
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate used for training."
)

parser.add_argument(
    "--epochs",
    type=int,
    default=2,
    help="Number of epochs used for training."
)

parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Number of max training steps. Overrides epochs."
)

parser.add_argument(
    "--model_name",
    type=str,
    default="t5-small",
    help="Name of HF pretrained model."
)

parser.add_argument(
    "--name",
    type=str,
    help="Name of the experiment. This determines where the checkpoints are saved."
)

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    task_dataset = TaskDataset(args.dataset_dir, args.train_file, args.test_file, args.model_name)
    train_dataset, test_dataset = task_dataset.get_dataset()
    
    arguments = TrainingArguments(
        output_dir="checkpoints/"+args.name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=5000,
        eval_steps=5000,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        seed=224
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

if __name__=="__main__":
    main(parser.parse_args())