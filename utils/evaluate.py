from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
from rouge import Rouge
import torch
from tqdm import tqdm
from typing import List


def sample(model: torch.nn.Module, tokenizer, prompt: str, **kwargs):
    prompt_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        prompt_tensor,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        **kwargs,
    )
    completion = tokenizer.decode(output[0][prompt_tensor.shape[1]:], skip_special_tokens=True)
    
    del prompt_tensor, output
    torch.cuda.empty_cache()

    return completion


def evaluate(model: torch.nn.Module, tokenizer, eval_dataset, *, min_new_tokens: int = 2, max_new_tokens: int = 100):
    completions: List[str] = []
    answers: List[str] = []
    model.eval()
    with torch.no_grad():
        for prompt, answer in tqdm(eval_dataset):
            if answer != "":
                completions.append(sample(model, tokenizer, prompt, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens))
                answers.append(answer)

    return {
        "accuracy score": accuracy_score(answers, completions),
        "bleu score": corpus_bleu([[ans.split()] for ans in answers], [comp.split() for comp in completions]),
        "rouge": Rouge().get_scores(completions, answers, avg=True)["rouge-1"]["f"],
    }
