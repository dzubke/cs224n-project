from enum import Enum
from typing import Dict, List, Union

TASK_ID = str
TASK_INSTANCE = Dict[str, Union[str, List[str]]]
TASK_INSTANCES = List[TASK_INSTANCE]
TASK_EXAMPLE = List[Dict[str, str]]
FILE_CONTENTS = Dict[str, Union[List[str], TASK_EXAMPLE, TASK_INSTANCES]]
TASK_FILENAME = str
DATASET = Dict[TASK_FILENAME, FILE_CONTENTS]


class SupNatKeys(Enum):
    CONTRIBUTORS = "Contributors"
    SOURCE = "Source"
    URL = "URL"
    CATEGORIES = "Categories"
    REASONING = "Reasoning"
    DEFINTION = "Definition"
    INPUT_LANGUAGE = "Input_language"
    OUTPUT_LANGUAGE = "Output_language"
    INSTRUCTION_LANGUAGE = "Instruction_language"
    DOMAINS = "Domains"
    POSITIVE_EXAMPLES = "Positive Examples"
    NEGATIVE_EXAMPLES = "Negative Examples"
    INSTANCES = "Instances"
    INSTANCE_LICENSE = "Instance License"


class ExampleKeys(Enum):
    INPUT = "input"
    OUTPUT = "output"
    EXPLANATION = "explanation"


class InstanceKeys(Enum):
    ID = "id"
    INPUT = "input"
    OUTPUT = "output"
