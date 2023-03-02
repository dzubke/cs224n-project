from src.data_utils.load_data import load_tasks_set
import pdb
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

def encode_input_def_pos1(definition, input):
    return "Definition: {} | Input: {}".format(definition, input)

class TaskDataset:
    def __init__(self, train_tasks_path, test_tasks_path, tokenizer="t5-small"):
        self.train_tasks_set = load_tasks_set(train_tasks_path)
        # TODO(hong): Use custom test tasks for evaluation.
        self.test_tasks_set = load_tasks_set(test_tasks_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=512)
    

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
        dataset = load_dataset("Muennighoff/natural-instructions")
        # TODO: replace HF dataset by loading from csv. This dataset has a different split than what we want. 
        # We concatenate train and validation since we do our own filtering.
        # dataset_cc = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        # dataset.to_csv('test.csv')
        # pdb.set_trace()
        dataset_dict = DatasetDict(
            train=dataset['train'].filter(lambda example: example["task_name"] in self.train_tasks_set),
            test=dataset['validation'].filter(lambda example: example["task_name"] not in self.train_tasks_set)
        )
        dataset_dict['train'].to_csv('train.csv')
        dataset_dict['test'].to_csv('test.csv')
        pdb.set_trace()
        tokenized = dataset_dict.map(self._tokenize_input_and_target, batched=True, batch_size=4)
        return tokenized['train'], tokenized['test']

