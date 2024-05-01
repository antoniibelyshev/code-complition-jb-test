from typing import Tuple, List, Union
import json


def write_dataset(samples: List[Tuple[str, str]], filename: str) -> None:
    with open(filename, 'w') as f:
        f.write('\n'.join(f"{{prompt: {sample[0]}, answer: {sample[1]}}}" for sample in samples))


class CodeCompletionDataset:
    def __init__(self, prompts: List[str], answers: List[str], train: bool):
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
    
    @staticmethod
    def from_json(filename: str, *, train: bool):
        with open(filename, 'r') as f:
            data = json.load(f)

        return CodeCompletionDataset(data['prompts'], data['answers'], train)
