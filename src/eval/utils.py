import json

# def trim_xlingual(reference_path):
#     examples = load_jsonl(reference_path)

#     trimmed_examples = []
#     for ex in examples:
#         if ex['


def load_jsonl(jsonl_path):

    output = []
    with open(jsonl_path, "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        output.append(json.loads(json_str))

    return output
