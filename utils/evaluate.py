from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from typing import List


def evaluate(model: torch.nn.Module, tokenizer, eval_dataset, *, max_new_tokens: int = 100):
    completions: List[str] = []
    answers: List[str] = []
    model.eval()
    with torch.no_grad():
        for prompt, answer in tqdm(eval_dataset):
            prompt_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
                bos_token_id=model.config.bos_token_id,
            )
            completions.append(tokenizer.decode(output[0], skip_special_tokens=True))

            answers.append(answer)

    return {
        "accuracy score": accuracy_score(answers, completions),
        "bleu score": corpus_bleu([comp.split() for comp in completions], [[ans.split()] for ans in answers]),
    }
