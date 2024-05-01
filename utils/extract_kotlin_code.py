import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from kopyt import Parser, node


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
    for filename in tqdm(kotlin_files[:8300] + kotlin_files[8400:]):
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

    print("Writing datasets...")
    return prompts, answers
