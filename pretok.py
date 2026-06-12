#!/usr/bin/env python3
import glob
import sys
import json

from datasets import Dataset, load_dataset
from collections import namedtuple

from pyarrow import parquet as pq

from aux import log
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


def data_sanity_check_and_len(path, cmd_args, proc_nums):
    files = glob.glob(path)
    lens = [pq.read_metadata(f).num_rows for f in files]

    assert all(e == lens[0] for e in lens), "Not all training data files have the same number of rows"

    assert len(files) % proc_nums.num_proc == 0, "Number of files is not divisible by number of processes"

    total_lens = sum(lens)

    nr_batches = total_lens // cmd_args.batch_size

    assert nr_batches * cmd_args.batch_size == total_lens, "batch arithmetics is failing us"

    return nr_batches * cmd_args.epochs


def load_training_data(path, cmd_args, proc_nums):
    #proc_nums.proc_idx
    #proc_nums.num_proc

    full_path = path + "/chunk*.parquet"

    nr_batches = data_sanity_check_and_len(full_path, cmd_args, proc_nums)
    if proc_nums.proc_idx == 0:
        log(f"Number of batches for {cmd_args.epochs} epochs: {nr_batches}")

    dataset = load_dataset("parquet", data_files=full_path, split="train", streaming=True)

    # Shard the dataset across your GPUs
    dataset = dataset.shard(num_shards=proc_nums.num_proc, index=proc_nums.proc_idx)

    # Shuffle locally within a buffer (mandatory for streaming to ensure local randomness)
    dataset = dataset.shuffle(buffer_size=10000, seed=42069)

    if cmd_args.epochs > 1:
        dataset = dataset.repeat(cmd_args.epochs)

    return dataset, nr_batches


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
