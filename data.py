#!/usr/bin/env python3
import promptops

import json, ijson
import sys

from random import shuffle

from torch.utils.data import Dataset as TorchDataset, IterableDataset

from aux import log


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


def prep_tokenized_prompt_from_entry(entry, selfx):
    # Return plain Python lists; let the collator pad & build labels.

    prompt = promptops.prep_prompt(entry, selfx.prompt_format)
    result = tokenize_str(selfx.tokenizer, prompt)
    result['special_tokens_mask'] = [False] * len(result['input_ids'])
    if selfx.sft_delim is not None:
        delim_id = selfx.tokenizer.convert_tokens_to_ids(selfx.sft_delim)
        delim_idx = result['input_ids'].index(delim_id)
        result['special_tokens_mask'][:delim_idx + 1] = [True] * (delim_idx + 1)

    elif selfx.sft_output_field is not None:
        no_output_prompt = promptops.prep_prompt({**entry, selfx.sft_output_field: ''}, selfx.prompt_format)
        no_output_prompt_tok = tokenize_str(selfx.tokenizer, no_output_prompt)
        len_to_mask = len(no_output_prompt_tok['input_ids'])
        result['special_tokens_mask'][:len_to_mask] = [True] * len_to_mask

    return result


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
        return prep_tokenized_prompt_from_entry(self.texts[idx], self)


"""
Go through texts iteratively without loading into memory,
returning tokenized tensors for readily formed prompts.
"""
class LazyTokenizingIterDataset(IterableDataset):
    def __init__(self, path, tokenizer, max_length=512, prompt_format="raw", sft_delim=None, sft_output_field=None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format
        self.sft_delim = sft_delim
        self.sft_output_field = sft_output_field

        self.ijson_iter = None

        self.data_len = self._get_data_len()

    def _get_data_len(self):
        result = 0

        with open(self.path, "r") as fh0:
            for _ in ijson.items(fh0, "item"):
                result += 1

        return result

    def __len__(self):
        return self.data_len

    def __iter__(self):
        fh = open(self.path, "r")
        self.ijson_iter = ijson.items(fh, "item")

        return self

    def __next__(self):
        #Return plain Python lists; let the collator pad & build labels.
        entry = next(self.ijson_iter)
        return prep_tokenized_prompt_from_entry(entry, self)


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

        prompt = promptops.prep_prompt(entry, self.prompt_format, inference=True, tok=self.tokenizer)
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



def load_training_data(path, tokenizer, cmd_args):
    if cmd_args.streamtrain:
        train_set_iter = LazyTokenizingIterDataset(path, tokenizer,
                                               cmd_args.max_length,
                                               cmd_args.prompt_format,
                                               cmd_args.sft_delim,
                                               cmd_args.sft_output_field)
    else:
        with open(path, "r") as f:
            data = json.load(f)

        train_set_iter = LazyTokenizingDataset(data, tokenizer,
                                               cmd_args.max_length,
                                               cmd_args.prompt_format,
                                               cmd_args.sft_delim,
                                               cmd_args.sft_output_field)

    return train_set_iter


if __name__ == '__main__':
    all_data = []

    for input_file in sys.argv[1:]:
        log(f"Processing {input_file}")
        with open(input_file, "r") as f:
            this_data = json.load(f)
            all_data += this_data

    log(f"Shuffling {len(all_data)} entries")
    shuffle(all_data)

    log(f"Saving")
    json.dump(all_data, sys.stdout)
