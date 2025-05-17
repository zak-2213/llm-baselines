from contextlib import nullcontext
from data.utils import get_dataloader

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import tiktoken

import torch
import torch.nn.functional as F
import wandb
import time 
import itertools
import copy
import random
import os
import numpy as np
from .utils import eval, get_batch, save_checkpoint

def train_base(model, opt, data, data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend,extra_args, itr=0,rng_state_dict=None):
    from optim.finetune import run_finetune
    print("Pretraining complete. Calling finetuning function...")
    run_finetune(model, checkpoint=ckpt_path)