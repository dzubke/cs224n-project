from src.data_utils.load_data import load_tasks_set
import pdb
from datasets import load_dataset, DatasetDict, concatenate_datasets, Features, Value
from transformers import AutoTokenizer

def encode_input_def_pos1(definition, input):
    return f"Definition: {definition}\nNow complete the following example−\ninput: {input}\noutput: "

class TaskDataset:
    def __init__(self,  dataset_dir, train_file, test_file, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_model_length=512)
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file

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
        # Save space and remove these columns for now.
        # First 100 instances in each task have eval=true.
        test_data = dataset['test'].filter(lambda e: e['eval'] == True)
        test_data = test_data.remove_columns(['eval', 'pos_0_input', 'pos_0_output', 'pos_0_explanation', 'neg_0_input', 'neg_0_output', 'neg_0_explanation', 'pos_1_input', 'pos_1_output', 'pos_1_explanation', 'neg_1_input', 'neg_1_output', 'neg_1_explanation'])
        dataset_dict = DatasetDict(train=train_dataset, test=test_data)
        tokenized = dataset_dict.map(self._tokenize_input_and_target, batched=True, num_proc=6)
        return tokenized['train'], tokenized['test']
