import sys
import csv
import itertools
from typing import *

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

WINDOW_SIZE = 1024
STEP_SIZE = int(WINDOW_SIZE / 2)

def sliding(iterable, window_size, step_size):
    assert step_size <= window_size
    its = []
    n = len(iterable)
    for i in range(window_size):
        if i < n:
            its.append(iterable[i:])
    previous = None
    for i, group in enumerate(zip(*its)):
        if i % step_size == 0:
            yield group
            previous = group
    remainder = i % step_size
    if remainder:
        yield tuple(iterable[-window_size:])

def query_text(text):
    tokens = tokenizer.encode(text)
    return long_query(tokens)

def long_query(tokens, step_size=STEP_SIZE, window_size=WINDOW_SIZE):
    first_window = True
    for window in tqdm.tqdm(sliding(tokens, window_size, step_size)):
        if first_window:
            yield from query_tokens(window)
            first_window = False
        else:
            yield from list(query_tokens(window))[-step_size:]

def query_tokens(tokens):
    token_str = list(map(tokenizer.decode, tokens))
    input = torch.LongTensor(tokens)
    with torch.no_grad():
        lnps = model(input).logits.log_softmax(-1)[:, input].diag(1)
    yield {
        'token': token_str[0],
        'logprob': float('nan'),
    }
    for token, lnp in zip(token_str[1:], lnps):
        yield {
            'token': token,
            'logprob': lnp.item(),
        }

def batch_query(input):
    with torch.no_grad():
        return model(input).logits.log_softmax(-1).gather(-1, input[:, 1:, None]).squeeze(-1)

def conditional_logp_single_token(context: str,
                                  completions: Iterable[str],
                                  sep=" "):
    tokens = [sep.join([context, completion]) for completion in completions]
    token_ids = torch.LongTensor([tokenizer.encode(token) for token in tokens])
    context_input = token_ids[0, :-1]
    completion_input = token_ids[:, -1]
    with torch.no_grad():
        lnp = model(context_input).logits.log_softmax(-1)[-1, completion_input]
    return lnp.log_softmax(-1)

def conditional_logp(context: str,
                     completions: Iterable[str],
                     padding=tokenizer.eos_token_id,
                     sep=" "):
    """ Get p(completion | context) normalized among completions """
    tokens = [sep.join([context, completion]) for completion in completions]
    input = torch.LongTensor(tokenize_with_padding(tokens, padding=padding))
    lnp = batch_query(input)
    mask = input[:, 1:] != padding
    total_lnp = (mask * lnp).sum(-1)
    return total_lnp.log_softmax(-1)

def _experimental_conditional_logp(context, completions, padding=tokenizer.eos_token_id):
    """ Get p(completion | context) normalized among completions. """
    # TODO: Find a way to do this in a way that save computation using past_key_values
    *context_tokens, final_context_token = tokenizer.encode(context)
    completion_tokens = torch.LongTensor(tokenize_with_padding(completions, padding=padding))
    full_tokens = torch.LongTensor([
        [final_context_token] + completion
        for completion in completion_tokens
    ])
    context_output = model(torch.LongTensor(context_tokens).unsqueeze(0))
    keys = replicate(context_output.past_key_values, len(completions))
    logits = model(tokens, past_key_values=keys).logits # shape B x L x V
    lnp = logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    mask = completion_tokens != padding
    total_lnp = (mask * lnp).sum(-1)
    return total_lnp.log_softmax(-1)

def tokenize_with_padding(strings: Iterable[str], padding=tokenizer.eos_token_id):
    tokens = list(map(tokenizer.encode, strings))
    lens = list(map(len, tokens))
    maxlen = max(lens)
    padded = [token + [padding]*(maxlen-len(token)) for token in tokens]
    return padded

def write_dicts(file, lines):
    lines_iter = iter(lines)
    first_line = next(lines_iter)
    writer = csv.DictWriter(file, first_line.keys())
    writer.writeheader()
    writer.writerow(first_line)
    for line in lines_iter:
        writer.writerow(line)

def main(text):
    write_dicts(sys.stdout, query_text(text))

if __name__ == '__main__':
    main(*sys.argv[1:])
    
    

