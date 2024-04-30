from typing import List
from dataclasses import dataclass
from collections import namedtuple
from tqdm import tqdm
from kopyt import Parser, node
import os
import fnmatch


@dataclass
class Function:
    definition: str
    prompt: str
    body: str

    def __init__(self, declaration: node.FunctionDeclaration):
        self.definition = str(declaration)
        params = [f"{param.name}: {param.type}" for param in declaration.parameters]
        
        self.prompt = f"fun {declaration.name}({', '.join(params)}): {declaration.type} "
        self.body = str(declaration.body)


def find_kotlin_files(directory):
    kotlin_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, '*.kt'):
                kotlin_files.append(os.path.join(root, file))
    return kotlin_files


def find_functions(kotlin_code: str) -> List[Function]:
    try:
        declarations = Parser(kotlin_code).parse().declarations
    except:
        return []
    res = [Function(declaration) for declaration in declarations if isinstance(declaration, node.FunctionDeclaration)]
    return res


if __name__ == "__main__":
    kotlin_files = find_kotlin_files("./kotlin/")

    functions: List[Function] = []
    for filename in tqdm(kotlin_files):
        with open(filename) as f:
            functions.extend(find_functions("".join(f.readlines())))
