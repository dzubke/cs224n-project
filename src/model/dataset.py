from src.data_utils.load_data import load_tasks_set
import pdb
from datasets import load_dataset, DatasetDict, concatenate_datasets, Features, Value
from transformers import AutoTokenizer

def encode_input_def_pos1(definition, input):
    return "Definition: {} | Input: {}".format(definition, input)

class TaskDataset:
    def __init__(self,  dataset_dir, train_file, test_file, tokenizer="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_model_length=512)
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file

    def output_csv(self):
        """Loads full training data and filters from the tasks that we specified."""
        train_path = self.dataset_dir + "/train_tasks.txt"
        test_path = self.dataset_dir + "/test_tasks.txt"
        train_set = load_tasks_set(train_path)
        test_set = load_tasks_set(test_path)

        # TODO: switch this out for our own file?
        dataset = load_dataset("Muennighoff/natural-instructions")
        
        # Combine all datasets from this HF datasets.
        all_datasets = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        # Filter for training set.
        filtered_train = all_datasets.filter(lambda example: example['task_name'] in train_set)
        print("Filtered train dataset num:{}".format(filtered_train.num_rows))
        # Make train and test exclusive.
        filtered_test = all_datasets.filter(lambda example: example['task_name'] in test_set and example['task_name'] not in train_set)
        print("Filtered test dataset num:{}".format(filtered_test.num_rows))

        filtered_train.to_csv(self.dataset_dir+"/train.csv")
        filtered_test.to_csv(self.dataset_dir+"/test.csv")

        with open(self.dataset_dir+"/summary.txt", 'w') as f:
            f.write("Total train examples: {}\n".format(dataset['train'].num_rows))
            f.write("Total test examples: {}\n".format(dataset['validation'].num_rows))
            f.write("\n")
            f.write("Num train tasks: {}\n".format(len(train_set)))
            f.write("Num examples: {}\n".format(filtered_train.num_rows))
            f.write("\n")
            f.write("Num test tasks: {}\n".format(len(test_set)))
            f.write("Num examples: {}\n".format(filtered_test.num_rows))
            

    def _tokenize_input_and_target(self, examples):
        input_strings = [encode_input_def_pos1(x,y) for x,y in zip(examples['definition'],examples['inputs'])]
        targets = examples['targets']
        # Encode the inputs
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding='max_length', truncation=True)
        # Encode the labels
        labels = self.tokenizer(targets, return_tensors="pt", padding='max_length', truncation=True).input_ids
        # Set loss to -100, which is ignored by CrossEntropyLoss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels
        return inputs


    def get_dataset(self):
        context_feat = Features({'task_name': Value(dtype='string', id=None), 
        'id': Value(dtype='string', id=None),
        'definition': Value(dtype='string', id=None),
        'inputs': Value(dtype='string', id=None),
        'targets': Value(dtype='string', id=None)
                                 })
        train_dataset = load_dataset('csv', data_files=self.dataset_dir+"/"+self.train_file, features=context_feat, on_bad_lines='skip')['train']
        test_dataset = load_dataset('csv', data_files=self.dataset_dir+"/"+self.test_file, features=context_feat, on_bad_lines='skip')['train']
        dataset_dict = DatasetDict(train=train_dataset, test=test_dataset)
        tokenized = dataset_dict.map(self._tokenize_input_and_target, batched=True, batch_size=4, num_proc=4)
        return tokenized['train'], tokenized['test']

