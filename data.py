#!/usr/bin/env python3
import promptops

import json
import sys

import pyarrow.parquet as pq
import glob

from torch.utils.data import Dataset as TorchDataset

from aux import log
from convdata import file_to_idx_name
from datasets import load_dataset


def tokenize_str(tokenizer, entry, add_eos=True, max_len=3000, for_inf=False):
    if for_inf:
        tokens = tokenizer(
            entry,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )
    else:
        tokens = tokenizer(
            entry,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True
        )

    if add_eos:
        tokens['attention_mask'].append(1)
        tokens['input_ids'].append(tokenizer.eos_token_id)

    return tokens


def prep_tokenized_prompt_from_entry(entry, selfx, tokenizr):
    # Return plain Python lists; let the collator pad & build labels.

    #try:
    prompt = promptops.prep_prompt(entry, selfx.prompt_format)
    result = tokenize_str(tokenizr, prompt)
    result['special_tokens_mask'] = [False] * len(result['input_ids'])
    if selfx.sft_delim is not None:
        delim_id = tokenizr.convert_tokens_to_ids(selfx.sft_delim)
        delim_idx = result['input_ids'].index(delim_id)
        result['special_tokens_mask'][:delim_idx + 1] = [True] * (delim_idx + 1)

    elif selfx.sft_output_field is not None:
        no_output_prompt = promptops.prep_prompt(data={**entry, selfx.sft_output_field: ''},
                                                 prompt_format=selfx.prompt_format)
        no_output_prompt_tok = tokenize_str(tokenizr, no_output_prompt)
        len_to_mask = len(no_output_prompt_tok['input_ids'])
        result['special_tokens_mask'][:len_to_mask] = [True] * len_to_mask

    return result

    #except:
    #    log("Broken data entry, returning a dummy instead")
    #    prompt = "dummy"
    #    result = tokenize_str(selfx.tokenizer, prompt)
    #    return result




"""
Load texts into memory and allow to loop through it,
returning tokenized tensors.
"""
class LazyTokenizingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512, prompt_format="raw", sft_delim=None, sft_output_field=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.sft_delim = sft_delim
        self.sft_output_field = sft_output_field

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return prep_tokenized_prompt_from_entry(self.texts[idx], self, self.tokenizer)


"""
Go through texts iteratively without loading into memory,
returning tokenized tensors for readily formed prompts.
"""
class LazyTokenizingIterDataset(TorchDataset):
    def __init__(self, path, tokenizer, max_dist=10000, max_length=512,
                 prompt_format="raw", sft_delim=None, sft_output_field=None, proc_nums=None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.sft_delim = sft_delim
        self.sft_output_field = sft_output_field
        self.max_dist = max_dist

        self.d_iter = None
        self.proc_nums = proc_nums

        self.data_len = self._get_data_len()

        self._curr_idx = 1e400

        self._restart_iters()

    def _get_this_shard_name(self, shard_idx=None):
        if shard_idx is None:
            shard_idx = self.proc_nums.proc_idx
        return file_to_idx_name(self.path, shard_idx)

    def _get_data_len(self):
        result = 0
        if self.proc_nums.proc_idx == 0:
            log("Computing length")

        for i in range(self.proc_nums.num_proc):
            with open(self._get_this_shard_name(shard_idx=i), "r") as fh0:
                for _ in fh0:
                    result += 1

        if self.proc_nums.proc_idx == 0:
            log(f"Length computed: {result}")

        return result

    def __len__(self):
        return self.data_len

    def _restart_iters(self):
        if self.proc_nums.proc_idx == 0:
            log("Restarting iterator")

        self.d_iter = open(self._get_this_shard_name(), "r")

        self._curr_idx = -1

    def __getitem__(self, idx):
        if self._curr_idx > idx:
            self._restart_iters()

        #assert idx % self.proc_nums.num_proc == self.proc_nums.proc_idx, f"MESS IN THREADS ({idx} % {self.proc_nums.num_proc} != {self.proc_nums.proc_idx})"

        line_idx = idx // self.proc_nums.num_proc

        assert self._curr_idx == line_idx - 1, "LINES SKIPPED"

        self._curr_idx += 1
        item_rawstr = next(self.d_iter)
        item = json.loads(item_rawstr)

        # !!! TODO_for_later: if it is too long or etc, then we skip it

        if item is None:
            raise Exception(f"This should not have happened: {self._curr_idx}, {idx} ({self.proc_nums.proc_idx})")

        result = prep_tokenized_prompt_from_entry(item, self, self.tokenizer)

        return result


class LazyTokenizingInferenceDataset(TorchDataset):
    def __init__(self, texts, tokenizer, prompt_format, max_length=512, debug=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.debug = debug

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        entry = self.texts[idx]

        prompt = promptops.prep_prompt(entry, self.prompt_format, inference=True)
        result = tokenize_str(self.tokenizer, prompt, add_eos=False, for_inf=True)

        if self.debug:
            log(f"Input: {prompt}")
            log(f"Tokenized: {result}")

        return result, prompt, entry


def read_input(path, formt):
    if path is None:
        log("Reading from STDIN")
        fh = sys.stdin
    else:
        fh = open(path, 'r')

    if formt == promptops.PF_RAW:
        result = [fh.read()]
    elif formt == promptops.PF_RAWLINES:
        result = fh.readlines()
    else:
        result = json.load(fh)

    return result


def get_data_loader(path, prompt_format, tokenizer, debug=False):
    inputs = read_input(path, prompt_format)

    dataset = LazyTokenizingInferenceDataset(inputs, tokenizer, prompt_format, debug=debug)

    """
    data_coll = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,  # helps performance; set None if you prefer exact lengths
    )

    data_loader = DataLoader(dataset, collate_fn=data_coll, batch_size=1)
    """

    return dataset


def data_sanity_check_and_len(path, cmd_args, proc_nums):
    files = glob.glob(path)
    lens = [pq.read_metadata(f).num_rows for f in files]

    assert all(e == lens[0] for e in lens), "Not all training data files have the same number of rows"

    assert len(files) % proc_nums.num_proc == 0, "Number of files is not divisible by number of processes"

    total_lens = sum(lens)

    nr_batches = total_lens // cmd_args.batch_size

    assert nr_batches * cmd_args.batch_size == total_lens, "batch arithmetics is failing us"

    return nr_batches * cmd_args.epochs

def load_training_data_new(path, cmd_args, proc_nums):
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
    dataset = dataset.shuffle(buffer_size=10000, seed=18736)

    if cmd_args.epochs > 1:
        dataset = dataset.repeat(cmd_args.epochs)

    def process_and_tokenize(entry):
        #return prep_tokenized_prompt_from_entry(entry, cmd_args, tokenizer)
        return promptops.prep_prompt(entry, cmd_args.prompt_format)

    return dataset.map(process_and_tokenize), nr_batches

def load_training_data(path, tokenizer, cmd_args, proc_nums):

    if cmd_args.streamtrain:
        train_set_iter = LazyTokenizingIterDataset(path, tokenizer,
                                               cmd_args.batch_size+3,
                                               cmd_args.max_length,
                                               cmd_args.prompt_format,
                                               cmd_args.sft_delim,
                                               cmd_args.sft_output_field, proc_nums=proc_nums)
    else:
        with open(path, "r") as f:
            data = json.load(f)

        train_set_iter = LazyTokenizingDataset(data, tokenizer,
                                               cmd_args.max_length,
                                               cmd_args.prompt_format,
                                               cmd_args.sft_delim,
                                               cmd_args.sft_output_field)

    return train_set_iter

