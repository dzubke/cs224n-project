from src.data_utils.load_data import load_tasks_set
import pdb
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

def encode_input_def_pos1(definition, input):
    return "Definition: {} | Input: {}".format(definition, input)

class TaskDataset:
    def __init__(self,  dataset_dir, train_file, test_file, tokenizer="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=512)
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file


    def _tokenize_input_and_target(self, examples):
        input_strings = [encode_input_def_pos1(x,y) for x,y in zip(examples['definition'],examples['inputs'])]
        targets = examples['targets']
        # Encode the inputs
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        # Encode the labels
        labels = self.tokenizer(targets, return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids
        # Set loss to -100, which is ignored by CrossEntropyLoss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels
        return inputs


    def get_dataset(self):
        train_dataset = load_dataset('csv', data_files=self.dataset_dir+"/"+self.train_file)['train']
        test_dataset = load_dataset('csv', data_files=self.dataset_dir+"/"+self.test_file)['train']
        dataset_dict = DatasetDict(train=train_dataset, test=test_dataset)
        tokenized = dataset_dict.map(self._tokenize_input_and_target, batched=True, batch_size=4)
        return tokenized['train'], tokenized['test']

