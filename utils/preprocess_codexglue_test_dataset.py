import json
from dataset import write_dataset
from typing import List, Tuple


replacements = {
    '<EOL>': '\n',
    '<INDENT>': '    ',
    '<DEDENT>': '',
    '<STR_LIT>': '',
    '<NUM_LIT>': '',
}

samples: List[Tuple[str, str]] = []
with open("codexglue_method_generation/test.jsonl") as f:
    for line in f:
        data = json.loads(line)

        signature = data['signature']
        docstring = data['docstring']

        samples.append(("\n    ".join([signature, docstring, ""]), data['body']))


write_dataset(samples, "./data/test_python_dataset.json")
