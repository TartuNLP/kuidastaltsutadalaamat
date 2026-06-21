#!/usr/bin/env python3
from datetime import datetime, timedelta

from aux import log, CmdlineArgs, env_stuff
from modelops import load_model, load_tokenizer
from pretok import load_training_data_pq_pretok
from data import load_training_data_jsonl

import subprocess

import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

import accelerate

accelerate.optimizer.AcceleratedOptimizer.defaults = property(
    lambda self: self.optimizer.defaults if hasattr(self.optimizer,
                                                    "defaults") else self.optimizer.optimizer.defaults
)

from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    logging,
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from collections import namedtuple

MEM_CHECK_KAMIKAZE = False

"""
1/3 This simply reads in command-line arguments 
"""

def _cmdline_args(acc):
    global MEM_CHECK_KAMIKAZE
    description = """Train or tune decoder models"""

    result = CmdlineArgs(description,
                         pos_arg_list=["mdl_id", "save_location", "train_file"],
                         pos_arg_types=[str, str, str],
                         kw_arg_dict={ "continue_training": False, "save_steps": 100, "lr": 1.5e-5,
                            "batch_size": 1024, "nr_sents_per_gpu": 4, "log_steps": 1, "epochs": 4,
                            "max_length": 4096,
                            "sharing": "none",
                            "prompt_format": "none",
                            "sft_delim": "none",
                            "sft_output_field": "none",
                            "gradckpt": False,
                            "debug": False,
                            "memcheckkamikaze": False,
                            "pretok": False,
                            "streamtrain": False})

    # if the directory args.save_location already exists, raise an exception:
    #if not result.continue_training and os.path.exists(result.save_location):
    #    raise Exception(f"Path '{result.save_location}' exists, won't overwrite - did you mean to set continue_training=True?.")

    if result.nr_sents_per_gpu == 0:
        result.nr_sents_per_gpu = result.batch_size

    if result.memcheckkamikaze:
        MEM_CHECK_KAMIKAZE = True

    log(f"Launched as {result}", accelerator=acc)

    return result


"""
2/3 This here is used in training in order to report timing and predictions 
"""

KamikazeException = Exception

class StepTimerCallback(TrainerCallback):
    def __init__(self):
        self._step_start = None
        self.lengths = []
        self.abs_start = datetime.now()

        self.actual_first_step = None

        self.zero = self.abs_start - self.abs_start

    # called right before each training step
    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = datetime.now()

        model = kwargs['model']

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                log(f"PROBLEMUUU: Weights in {name} (shape {param.shape}) are already NaN BEFORE this step.")
                raise Exception("Pre-existing NaN weights")

    # called right after each training step
    def on_step_end(self, args, state, control, **kwargs):
        global MEM_CHECK_KAMIKAZE
        now = datetime.now()

        if self.actual_first_step is None:
            self.actual_first_step = state.global_step - 1

        elapsed = now - self._step_start
        tot_elapsed = now - self.abs_start
        self.lengths.append(elapsed)

        avg = sum(self.lengths, start=self.zero) / len(self.lengths)

        prediction = avg * (state.max_steps - state.global_step)

        log(f"step {state.global_step}/{state.max_steps} took {elapsed}, avg {avg}; approx {prediction} remaining")

        if MEM_CHECK_KAMIKAZE and state.global_step >= 13:
            rocm_output = subprocess.check_output(['rocm-smi'])

            print(rocm_output.decode('utf8'))

            log(f"memory measurement done!")

            raise KamikazeException

        model = kwargs['model']

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                log(f"PROBLEMIIII: Weights in {name} (shape {param.shape}) are NaN after this step.")
                raise Exception("Pre-existing NaN weights")


"""
3/3 Finally, the filling of TrainingArguments and the launching of Trainer:
"""

def get_deepspeed_conf(cmdline_args, accum_steps):
    if cmdline_args.sharing == "deepspeed":
        return {'deepspeed': {
            'train_batch_size': cmdline_args.batch_size,
            'train_micro_batch_size_per_gpu': cmdline_args.nr_sents_per_gpu,
            'gradient_accumulation_steps': accum_steps,
            "bf16": { "enabled": True },

            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": { "device": "none" },
                "allgather_partitions": True,
                "overlap_comm": True,
                "allgather_bucket_size": 50000000,
                "reduce_scatter": True,
                "reduce_bucket_size": 50000000,
                "contiguous_gradients": True,
                "communication_data_type": "fp32",
            },

            "gradient_clipping": 1.0,
            "steps_per_print": 20,
            "wall_clock_breakdown": False
        }}
    else:
        return {}


def _get_dec_layer_from_mdl(model_id):
    if "Tower-Plus" in model_id or "towerplus" in model_id:
        return "Gemma2DecoderLayer"
    elif "pertus" in model_id:
        return "ApertusDecoderLayer"
    else:
        return "LlamaDecoderLayer"


def get_fsdp_conf(cmdline_args):
    dec_layer_name = _get_dec_layer_from_mdl(cmdline_args.mdl_id)

    if cmdline_args.sharing == "fsdp":
        return {'fsdp': "shard_grad_op auto_wrap",
            'fsdp_config': {
                "use_orig_params": True,
                "sync_module_states": True,
                "forward_prefetch": True,
                "limit_all_gathers": True,
                # "reshard_after_forward": True,
                # "fsdp_min_num_params": 1e7,
                "transformer_layer_cls_to_wrap": [dec_layer_name],
                # DO NOT enable cpu_offload on LUMI unless desperate
            }}
    else:
        return {}

def get_training_args(cmdline_args, acc, total_batches):
    #auto_find_batch_size
    world_size = acc.num_processes

    assert cmdline_args.batch_size % (cmdline_args.nr_sents_per_gpu * world_size) == 0, \
        "Batch size must be divisible by the number of GPUs and nr of sents per GPU"

    accum_steps = cmdline_args.batch_size // (cmdline_args.nr_sents_per_gpu * world_size)
    acc.gradient_accumulation_steps = accum_steps

    log(f"Nr of processes (GPUs): {world_size}, "
        f"per-device batch: {cmdline_args.nr_sents_per_gpu}, "
        f"accum. steps: {accum_steps}", accelerator=acc)

    dpspd_conf = get_deepspeed_conf(cmdline_args, accum_steps)
    fsdp_conf = get_fsdp_conf(cmdline_args)

    tr_args = TrainingArguments(
        output_dir=cmdline_args.save_location,
        per_device_train_batch_size=cmdline_args.nr_sents_per_gpu,
        gradient_accumulation_steps=accum_steps,
        #num_train_epochs=cmdline_args.epochs,
        save_steps=cmdline_args.save_steps,
        save_total_limit=2,
        max_steps=total_batches,
        logging_steps=cmdline_args.log_steps,
        learning_rate=cmdline_args.lr,
        save_strategy="steps",
        disable_tqdm=True,
        report_to="none",
        lr_scheduler_type = "polynomial",
        weight_decay = 0.1,
        bf16=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        warmup_steps=500,
        #train_sampling_strategy="group_by_length",
        log_level="debug" if cmdline_args.debug else "passive",
        optim="adamw_torch",
        accelerator_config={ 'dispatch_batches': False },
        gradient_checkpointing=cmdline_args.gradckpt,
        #dataloader_persistent_workers=True
        **dpspd_conf,
        **fsdp_conf,
    )

    return tr_args


class NoShardTrainer(Trainer):
    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


import torch
import torch.nn.functional as F
import types


def safe_xielu_forward_disabl(self, x):
    x_fp32 = x.to(torch.float32)
    x_fp32 = torch.clamp(x_fp32, min=-50.0, max=50.0)

    # Note: adjust parameter names if they differ in modeling_apertus.py
    alpha_p = F.softplus(self.alpha_p.to(torch.float32))
    alpha_n = F.softplus(self.alpha_n.to(torch.float32))
    beta = 0.5

    pos_mask = (x_fp32 > 0)
    neg_mask = ~pos_mask

    out = torch.zeros_like(x_fp32)
    out[pos_mask] = alpha_p * (x_fp32[pos_mask] ** 2) + beta * x_fp32[pos_mask]

    x_neg = x_fp32[neg_mask]
    out[neg_mask] = alpha_n * (torch.exp(x_neg) - 1.0) - alpha_n * x_neg + beta * x_neg

    return out.to(x.dtype)

"""

class NoNanTrainer(NoShardTrainer):
    def compute_loss(self, model, inputs, **kwargs):

        maxval = inputs['input_ids'].max().item()

        if maxval >= 131072:
            log(f"PROBLEMX: input_ids contains {maxval} (max value)")

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                log(f"PROBLEMA: Weights in {name} (shape {param.shape}) are already NaN BEFORE this step.")
                raise Exception("Pre-existing NaN weights")

        result = super().compute_loss(model, inputs, **kwargs)

        if torch.isnan(result).any():
            log(f"PROBLEMA: loss {result}")
            log(f"PROBLEMA: inputs {inputs['input_ids']}")
            log(f"PROBLEMA: outputs {inputs['labels']}")
            raise Exception("NaN loss again")

        return result


    def training_step(self, model, inputs, *args, **kwargs):
        maxval = inputs['input_ids'].max().item()

        if maxval >= 131072:
            log(f"PROBLEMY: input_ids contains {maxval} (max value)")

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                log(f"PROBLEM: Weights in {name} (shape {param.shape}) are already NaN BEFORE this step.")
                log(f"max input: {maxval}")
                raise Exception("Pre-existing NaN weights")

        loss = super().training_step(model, inputs, *args, **kwargs)

        if torch.isnan(loss):
            log(f"PROBLEM: loss {loss}")
            log(f"PROBLEM: inputs {inputs['input_ids']}")
            log(f"PROBLEM: outputs {inputs['labels']}")
            raise Exception("NaN loss")

        return loss


class BatchTrackingTrainer(NoNanTrainer):
    def compute_loss(self, model, inputs, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            log(f"PROBLEMXXX: Logits contain NaN/Inf. Max val: {logits.max()}, Min val: {logits.min()}; all: {logits}")
            raise Exception("NaN logits")

        result = super().compute_loss(model, inputs, **kwargs)
        return result

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)

        log(f"BATCH_LOG_LOSS: {loss.item()}")

        return loss

def nan_hook(module, inp, out):
    if isinstance(out, torch.Tensor) and not out.is_floating_point():
        return
    if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
        raise RuntimeError(f"NaN/Inf in output of {module.__class__.__name__}")
"""

def simple_train(acc):
    cmd_args = _cmdline_args(acc)
    proc_nums = namedtuple("ProcNums",
                           ["proc_idx", "num_proc"])(acc.process_index, acc.num_processes)
    if cmd_args.debug:
        torch.autograd.set_detect_anomaly(True)

    device = None if cmd_args.sharing == "fsdp" else acc.device

    tokenizer = load_tokenizer(cmd_args.mdl_id, acc)
    log(f"Tokenized vocab size: {len(tokenizer)}", accelerator=acc)

    model = load_model(cmd_args.mdl_id, device, acc, attention="flash_attention_2")
    #for module in model.modules():
    #    if module.__class__.__name__ == "XIELUActivation":
    #        module.forward = types.MethodType(safe_xielu_forward, module)

    if cmd_args.gradckpt:
        model.gradient_checkpointing_enable()

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    log(f"Model embeddings: {model.get_input_embeddings().weight.shape}")

    log(f"Load data", accelerator=acc)
    if cmd_args.pretok:
        tokenized_train_data, total_batches = load_training_data_pq_pretok(cmd_args.train_file, cmd_args, proc_nums)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    else:
        tokenized_train_data, total_batches = load_training_data_jsonl(cmd_args.train_file, tokenizer, cmd_args, proc_nums)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=1,
            random_replace_prob=0,
            mask_replace_prob=0,
            pad_to_multiple_of=8,
        )

    log(f"Preparing to train with {total_batches} steps", accelerator=acc)

    clbks = [StepTimerCallback] if acc.is_main_process else []

    training_args = get_training_args(cmd_args, acc, total_batches)

    TrainerClass = NoShardTrainer # BatchTrackingTrainer if cmd_args.debug else NoNanTrainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=clbks,
    )

    #trainer._get_train_sampler = types.MethodType(lambda self, ds: SequentialSampler(ds), trainer)
    if cmd_args.debug:
        logging.set_verbosity_debug()

        #for layer in model.modules():
        #    layer.register_forward_hook(nan_hook)

    log(f"Starting training", accelerator=acc)
    trainer.train(resume_from_checkpoint=cmd_args.continue_training)

    log(f"All done!", accelerator=acc)


if __name__ == "__main__":
    env_stuff()

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])

    simple_train(accelerator)
