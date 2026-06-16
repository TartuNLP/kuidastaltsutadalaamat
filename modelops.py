import torch

from aux import log
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(mdl_id, device, accelerator=None, attention="flash_attention_2"):
    log(f"Load model", accelerator=accelerator)


    model = AutoModelForCausalLM.from_pretrained(mdl_id,
                                                 local_files_only=True,
                                                 device_map=None,
                                                 low_cpu_mem_usage=False,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation=attention)


    model.config.use_cache = False
    if device is not None:
        model = model.to(device)
        log(f"Model loaded on device: {model.device}.", accelerator=accelerator)

    return model


def load_tokenizer(mdl_id, accelerator=None, left_padding=False):
    log(f"Load tokenizer", accelerator=accelerator)
    tokenizer = AutoTokenizer.from_pretrained(mdl_id)
    tokenizer.padding_side = "left" if left_padding else "right"

    #tokenizer.pad_token = "<|reserved_special_token_100|>"
    tokenizer.mask_token = "<|reserved_special_token_130|>"

    return tokenizer
