import json
from typing import List, Tuple


def read_codexglue_test_data() -> Tuple[List[str], List[str]]:
    replacements = [
        ('<EOL>', '\n'),
        ('<INDENT>', '    '),
        ('<DEDENT>', ''),
        ('<STR_LIT>', ''),
        ('<NUM_LIT>', ''),
    ]

    prompts: List[str] = []
    answers: List[str] = []
    with open("codexglue_method_generation/test.jsonl") as f:
        for line in f:
            data = json.loads(line)

            signature = data['signature']
            docstring = data['docstring']
            prompt = "\n    ".join([signature, docstring, ""])
            prompts.append(prompt)

            answer = data['body']
            for symbol, replacement in replacements:
                answer = answer.replace(symbol, replacement)
            answers.append(answer)

    return prompts, answers
