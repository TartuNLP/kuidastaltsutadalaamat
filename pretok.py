#!/usr/bin/env python3

import sys
import json

from datasets import Dataset
from collections import namedtuple

from modelops import load_tokenizer
from promptops import prep_tokenized_prompt_from_entry, PF_SUURTOLK


def data_gen(filename, tokenizer, more_args):
    with open(filename, 'r') as fh:
        for line in fh:
            entry = json.loads(line)

            tokenized = prep_tokenized_prompt_from_entry(entry, more_args, tokenizer)

            labels = [
                -100 if m else t
                for t, m in zip(tokenized["input_ids"], tokenized["special_tokens_map"])
            ]

            yield { 'input_ids': tokenized["input_ids"],
                    'labels': labels,
                    'attention_mask': tokenized["attention_mask"] }


def jsonl_to_parquet(in_filename, out_filename, tokenizer, more_args):
    generator = lambda: data_gen(in_filename, tokenizer, more_args)

    dataset = Dataset.from_generator(generator)

    dataset.to_parquet(out_filename)


def cmdline():
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    tokenizer_id = sys.argv[3]

    try:
        prompt_format = sys.argv[4]
    except IndexError:
        prompt_format = PF_SUURTOLK

    try:
        sft_delim = sys.argv[5]
    except IndexError:
        sft_delim = "<|assistant_start|>"

    args = (namedtuple("CmdArgs",
                       "input_file output_file tok_id prompt_format sft_delim sft_output_field")
            (in_filename, out_filename, tokenizer_id, prompt_format, sft_delim, None))

    return args


def say_no_to_global_variables():
    cmdargs = cmdline()

    tokenizer = load_tokenizer(cmdargs.tok_id)

    jsonl_to_parquet(cmdargs.input_file, cmdargs.output_file, tokenizer, cmdargs)

if __name__ == '__main__':
    say_no_to_global_variables()

"""
import json
from datasets import Dataset
from transformers import AutoTokenizer
from promptops import prompt_builder_from_entry
from data import prep_tokenized_prompt_from_entry

# Configuration
tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
jsonl_file = "train_data/chunk_0.jsonl"
output_parquet = "train_data/chunk_0.parquet"

def gen():
    with open(jsonl_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Leverage your existing logic
            # Note: You'll want to modify your data.py function to return 'labels' 
            # instead of just the map, or calculate them here.
            tokenized = prep_tokenized_prompt_from_entry(entry, None, tokenizer)
            
            # Example label logic: mask everything except the assistant output
            tokenized["labels"] = [
                t if m == 1 else -100 
                for t, m in zip(tokenized["input_ids"], tokenized["special_tokens_map"])
            ]
            yield tokenized

dataset = Dataset.from_generator(gen)
dataset.to_parquet(output_parquet)
print(f"Successfully saved {output_parquet}")
"""