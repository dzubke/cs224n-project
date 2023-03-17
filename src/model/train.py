
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
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate used for training."
)

parser.add_argument(
    "--learning_rates",
    type=str,
    default="1e-4",
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

def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "rougeL", "goal": "maximize"},
        "early_terminate": {"type":"hyperband", "min_iter": 3},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 5e-6, "max": 1e-3},
            # "per_device_train_batch_size": {"values": [32]},
        },
    }

def compute_objective(metrics):
    return metrics['rougeL']

def model_init():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
    return model

def train(args, train_file, learning_rate):
    name = train_file.split("/")[-1].split(".")[0]
    description = f"{name}-{learning_rate}"
    run = wandb.init(reinit=True, name=description)
    print(f"training {description}")

    task_dataset = TaskDataset(args.dataset_dir, train_file, args.test_file, args.model_name)
    train_dataset, test_dataset = task_dataset.get_dataset()

    arguments = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{description}",
        learning_rate=learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=2500,
        eval_steps=2500,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="wandb",
        fp16=True,
        seed=224
    )
    trainer = Seq2SeqTrainer(
        model=None,
        model_init=model_init,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=task_dataset.compute_metrics,
    ) 
    trainer.train()
    run.finish()

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    model = model_init()
    for train_file in args.train_file.split(","):
        for lr in args.learning_rates.split(","):
            train(args, train_file, float(lr))

if __name__=="__main__":
    main(parser.parse_args())