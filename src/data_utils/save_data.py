import pdb
from datasets import load_dataset, DatasetDict

from src.model.dataset import TaskDataset

model_name = "t5-small"

if __name__=="__main__":
    task_dataset = TaskDataset(dataset_dir="", train_file="", test_file="", model_name=model_name)
    dataset = load_dataset("jayelm/natural-instructions")
    # First 100 instances in each task have eval=true.
    test_data = dataset['test'].filter(lambda e: e['eval'] == True)
    dataset_dict = DatasetDict(train=dataset['train'], test=test_data)
    dataset_dict['test'] = dataset_dict['test'].remove_columns(['eval', 'pos_0_input', 'pos_0_output', 'pos_0_explanation', 'neg_0_input', 'neg_0_output', 'neg_0_explanation', 'pos_1_input', 'pos_1_output', 'pos_1_explanation', 'neg_1_input', 'neg_1_output', 'neg_1_explanation'])
    dataset_dict['train'] = dataset_dict['train'].remove_columns(['eval', 'pos_0_input', 'pos_0_output', 'pos_0_explanation', 'neg_0_input', 'neg_0_output', 'neg_0_explanation', 'pos_1_input', 'pos_1_output', 'pos_1_explanation', 'neg_1_input', 'neg_1_output', 'neg_1_explanation'])
    pdb.set_trace()
    tokenized = dataset_dict.map(task_dataset._tokenize_input_and_target, batched=True, num_proc=4)
    pdb.set_trace()

    tokenized['train'].to_json('train.json')
    tokenized['test'].to_json('test.json')