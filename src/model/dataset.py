from src.data_utils.load_data import load_tasks_set
import pdb
import evaluate
from datasets import load_dataset, DatasetDict, concatenate_datasets, Features, Value
import numpy as np
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")

def encode_input_def_pos1(definition, input):
    return f"Definition: {definition}\nNow complete the following example−\ninput: {input}\noutput: "

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

class TaskDataset:
    def __init__(self,  dataset_dir, train_file, test_file, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_model_length=512)
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file
        self.metric = evaluate.load("rouge")

    def _tokenize_input_and_target(self, examples):
        input_strings = [encode_input_def_pos1(x,y) for x,y in zip(examples['definition'],examples['inputs'])]
        targets = examples['targets']
        # Encode the inputs
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        # Encode the labels
        labels = self.tokenizer(targets, return_tensors="pt", padding='max_length', truncation=True, max_length=128).input_ids
        # Set loss to -100, which is ignored by CrossEntropyLoss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels
        return inputs

    def get_dataset(self):
        dataset = load_dataset("jayelm/natural-instructions")
        if not self.train_file:
            dataset['train'] = dataset['train'].remove_columns(['eval', 'pos_0_input', 'pos_0_output', 'pos_0_explanation', 'neg_0_input', 'neg_0_output', 'neg_0_explanation', 'pos_1_input', 'pos_1_output', 'pos_1_explanation', 'neg_1_input', 'neg_1_output', 'neg_1_explanation'])            
            train_dataset = dataset['train']
        else:
            # Load custom training set.
            train_dataset = load_dataset('json', data_files=self.train_file)['train']
        # First 100 instances in each task have eval=true.
        test_data = dataset['test'].filter(lambda e: e['eval'] == True)
        # Save space and remove these columns for now.
        test_data = test_data.remove_columns(['eval', 'pos_0_input', 'pos_0_output', 'pos_0_explanation', 'neg_0_input', 'neg_0_output', 'neg_0_explanation', 'pos_1_input', 'pos_1_output', 'pos_1_explanation', 'neg_1_input', 'neg_1_output', 'neg_1_explanation'])
        dataset_dict = DatasetDict(train=train_dataset, test=test_data)
        tokenized = dataset_dict.map(self._tokenize_input_and_target, batched=True, num_proc=6)
        return tokenized['train'], tokenized['test']
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result