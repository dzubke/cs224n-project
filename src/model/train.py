
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import argparse
from src.model.dataset import TaskDataset
import wandb

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
    "--gradient_accumulation_steps",
    type=int,
    default=1,
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate used for training."
)

parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="linear",
    help="linear or constant"
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

def train(args, model, train_file):
    name = train_file.split("/")[-1].split(".")[0]
    description = f"{name}-{args.learning_rate}"
    run = wandb.init(reinit=True, name=description)
    print(f"training {description}")

    task_dataset = TaskDataset(args.dataset_dir, train_file, args.test_file, args.model_name)
    train_dataset, test_dataset = task_dataset.get_dataset()

    arguments = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{description}",
        optim="adamw_torch",
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=2500,
        eval_steps=2500,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="wandb",
        fp16=False,
        seed=224
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=task_dataset.compute_metrics,
    ) 
    trainer.train()
    run.finish()

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    print("Using device: ", device)
    for train_file in args.train_file.split(","):
        train(args, model, train_file)

if __name__=="__main__":
    main(parser.parse_args())