#!/usr/bin/env python3
import promptops

import json
import sys

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset

from aux import log
from convdata import file_to_idx_name

from promptops import tokenize_str, prep_tokenized_prompt_from_entry

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
class LazyTokenizingIterDataset(IterableDataset):
    def __init__(self, path, tokenizer, max_dist=10000, max_length=512,
                 prompt_format="raw", sft_delim=None, sft_output_field=None, proc_nums=None, debug=False):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.sft_delim = sft_delim
        self.sft_output_field = sft_output_field
        self.max_dist = max_dist
        self.debug = debug

        self.d_iter = None
        self.proc_nums = proc_nums

        self.data_len = self._get_data_len()

        #self._restart_iters()

    def _get_this_shard_name(self, shard_idx=None):
        if shard_idx is None:
            shard_idx = self.proc_nums.proc_idx
        return file_to_idx_name(self.path, shard_idx)

    def _get_data_len(self):
        if self.proc_nums.proc_idx == 0 and self.debug:
            log("Computing length")

        #for i in range(self.proc_nums.num_proc):
        result = 0
        with open(self._get_this_shard_name(shard_idx=self.proc_nums.proc_idx), "r") as fh0:
            for _ in fh0:
                result += 1

        if self.proc_nums.proc_idx == 0 and self.debug:
            log(f"Length computed: {result}")

        return result

    #def __len__(self):
    #    return self.data_len

    def ___restart_iters(self):
        if self.proc_nums.proc_idx == 0 and self.debug:
            log("Restarting iterator")

        self.d_iter = open(self._get_this_shard_name(), "r")

    def __iter__(self):
        with open(self._get_this_shard_name(), "r") as fh:
            for item_rawstr in fh:
                if self.debug:
                    log(f"PROMPT_LOG_START")

                item = json.loads(item_rawstr)

                if self.debug:
                    log(f"PROMPT_LOG_LOADED /// {str(item)[:200]}")

                if item is None:
                    raise Exception(
                        f"This should not have happened")

                result = prep_tokenized_prompt_from_entry(item, self, self.tokenizer)

                if self.debug:
                    log(f"PROMPT_LOG_TOKENIZED /// {str(result)[:200]}")

                yield result

    """
    def __getitem__(self, idx):
        if self._curr_idx > idx:
            self._restart_iters()

        #assert idx % self.proc_nums.num_proc == self.proc_nums.proc_idx, f"MESS IN THREADS ({idx} % {self.proc_nums.num_proc} != {self.proc_nums.proc_idx})"

        line_idx = idx // self.proc_nums.num_proc

        msg = f"LINES SKIPPED: {self._curr_idx + 1} should be equal to {line_idx} = {idx} // {self.proc_nums.num_proc}"
        assert self._curr_idx + 1 == line_idx, msg

        self._curr_idx += 1
        item_rawstr = next(self.d_iter)
        item = json.loads(item_rawstr)

        # !!! TODO_for_later: if it is too long or etc, then we skip it

        if item is None:
            raise Exception(f"This should not have happened: {self._curr_idx}, {idx} ({self.proc_nums.proc_idx})")

        result = prep_tokenized_prompt_from_entry(item, self, self.tokenizer)

        return result
"""

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


def load_training_data_old(path, tokenizer, cmd_args, proc_nums):

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

    nr_batches = train_set_iter.data_len * proc_nums.num_proc // cmd_args.batch_size
    log(f"nr_batches {nr_batches} = data_len {train_set_iter.data_len} * num_proc {proc_nums.num_proc} // batch_size {cmd_args.batch_size}")
    return train_set_iter, nr_batches

