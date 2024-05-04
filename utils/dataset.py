from typing import Tuple, List, Union
import torch
from tqdm import tqdm
from kopyt import Parser, node
import json
import os


class CodeCompletionDataset:
    def __init__(self, prompts: List[str], answers: List[str], *, train: bool):
        self.prompts = prompts
        self.answers = answers

        self.train = train

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> Union[str, Tuple[str, str]]:
        if self.train:
            return self.prompts[index] + self.answers[index]
        else:
            return self.prompts[index], self.answers[index]


class TrainDataset:
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding=True)
        res =  {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask'])
        }

        del text, encoding

        return res
    

def extract_kotlin_code(
        *,
        kotlin_directory: str = "./kotlin/",
) -> Tuple[List[str], List[str]]:
    print("Looking for kotlin files...")
    kotlin_files: List[str] = []
    for root, dirs, files in os.walk(kotlin_directory):
        for file in files:
            if file.endswith(".kt"):
                kotlin_files.append(os.path.join(root, file))

    print("Parsing functions in kotlin files...")
    prompts: List[str] = []
    answers: List[str] = []
    parse_errors_count = 0
    declaration_errors_count = 0
    # for filename in tqdm(kotlin_files[:8300] + kotlin_files[8400:]):
    for filename in tqdm(kotlin_files[:300]):
        with open(filename) as f:
            kotlin_code = f.read()

        try:
            declarations = Parser(kotlin_code).parse().declarations
        except:
            parse_errors_count += 1
            continue

        for declaration in declarations:
            if not isinstance(declaration, node.FunctionDeclaration):
                continue
            try:
                params = [f"{param.name}: {param.type}" for param in declaration.parameters]
                prompt = f"fun {declaration.name}({', '.join(params)}): {declaration.type} "
                answer = str(declaration.body)
                prompts.append(prompt)
                answers.append(answer)
            except:
                declaration_errors_count += 1

    print(f"parse errors count: {parse_errors_count}, declaration errors count: {declaration_errors_count}")
    print(f"total number of samples: {len(prompts)}")

    return prompts, answers


def read_codexglue_test_data(n: int = 20000) -> Tuple[List[str], List[str]]:
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
        for line in f.readlines()[:n]:
            data = json.loads(line)

            signature = data['signature']
            docstring = "\"\"\"\n    " + data['docstring'].replace("\n", "\n    ") + "\n    \"\"\""
            prompt = "\n    ".join([signature, docstring, ""])
            prompts.append(prompt)

            answer = data['body']
            for symbol, replacement in replacements:
                answer = answer.replace(symbol, replacement)
            answers.append(answer)

    return prompts, answers
