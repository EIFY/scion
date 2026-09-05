
# Modified from: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
from typing import Optional

import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for modded-nanogpt

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        bound = self.n_embd ** -0.5
        self.c_qkv = nn.Parameter(torch.empty(3, self.n_embd, self.n_embd).uniform_(-bound, bound))
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = F.linear(x, self.c_qkv.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.n_head, -1).chunk(3, dim=-2)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        # Uses scaled ReLU, `sqrt(2)*relu(x)`, as the basis
        x = (math.sqrt(2)*F.relu(x)).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying # CHANGE

    def forward(self, idx):
        # Just the output embeddings
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        return F.rms_norm(x, (x.size(-1),))

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

def _maybe_shuffle(original, seed):
    if seed is None:
        return original, seed
    shuffled = list(original)
    random.Random(seed).shuffle(shuffled)
    return shuffled, seed + 1

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = nstep_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
            nstep_total += (shard_ntok - 1) // (num_processes * B * T)
        self.ntok_total = ntok_total
        self.nstep_total = nstep_total

    def token_generator(self, seed=None):
        BT = self.B * self.T
        start = self.process_rank * BT
        global_bs = self.num_processes * BT
        while True:
            files, seed = _maybe_shuffle(self.files, seed)
            for f in files:
                tokens = _load_data_shard(f)
                offsets, seed = _maybe_shuffle(range(start, len(tokens) - global_bs, global_bs), seed)
                for offset in offsets:
                    buf = tokens[offset : offset + BT + 1]
                    buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
                    x = (buf[:-1]).view(self.B, self.T) # inputs
                    y = (buf[1:]).view(self.B, self.T) # targets
                    yield x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    name : Optional[str] = None
    input_bin : str = 'data/fineweb-edu100B/fineweb_edu_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb-edu100B/fineweb_edu_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 16 # batch size, in sequences, across all devices
    device_batch_size : int = 16 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    steps : int = 0 # number of iterations to run. Defaults to 1 epoch
    seed : Optional[int] = None # change to an int to shuffle files and offsets
    lr : float = 2 ** -12 * 50 # Max LR
    cos_power : float = 1.0 # power of the cosine LR decay, defaults to 1.0
    power : Optional[float] = None # power of the polynomial LR decay, defaults to None (cosine LR decay)
    corrected : bool = False
    c_sq : float = 5.79833984375 # (2 - 0.1) / (2 * 0.1) * 2 ** -12 * 50 ** 2
    wd : float = 1 / 50
    sign_lr : float = 2 ** -12 * 3000
    sign_wd : float = 1 / 3000
    grad_clip_norm : float = 1000000. # effectively no clipping
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485761 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons.
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    n_layer : int = 12
    n_head : int = 6 # set as n_embd/128 so head_dim is 128
    n_embd : int = 768
    momentum : float = 0.1
    mdc : float = 0.0 # Momentum decay constant
    end_c_sq_mul : float = 1.0
    cautious : bool = False
    nesterov : bool = False
    s_mo : Optional[float] = None # Momentum for sign paramters, defaults to momentum
    s_ne : Optional[bool] = None # Nesterov or not for sign parameters, defaults to nesterov
from datargs import parse

def norm_info(state_dict):
    count = 0
    hidden = 0.
    for n, p in state_dict.items():
        if n == '_orig_mod.transformer.wte.weight': # weight-tying this is identical to _orig_mod.lm_head.weight
            continue
        elif n == '_orig_mod.lm_head.weight':
            continue
        else:
            count += p.numel()
            hidden += torch.sum(p ** 2).item()
    return count, hidden

def perturb(state_dict, std):
    for n, p in state_dict.items():
        if n == '_orig_mod.transformer.wte.weight': # weight-tying this is identical to _orig_mod.lm_head.weight
            continue
        elif n == '_orig_mod.lm_head.weight':
            continue
        else:
            p += torch.randn_like(p) * std
    return

def main():

    args = parse(Hyperparameters)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    if master_process:
        print("======== Arguments ========")
        print(args)
        print("===========================")

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

    tokens_per_global_batch = B * T * ddp_world_size
    if not args.val_tokens:
        val_steps = val_loader.nstep_total
    else:
        # calculate the number of steps to take in the val loop.
        assert (args.val_tokens - 1) % tokens_per_global_batch == 0
        val_steps = (args.val_tokens - 1) // tokens_per_global_batch

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)

    print(args.name)
    resume = os.path.join("logs", args.name, "state_step030250.pt")
    ckpt = torch.load(resume, weights_only=False)
    state_dict = ckpt['model']
    count, norm_squared = norm_info(state_dict)
    print(f"{count=} {math.sqrt(norm_squared)=}")

    model.load_state_dict(state_dict)
    model = DDP(model, device_ids=[ddp_local_rank])
    
    perturbed_model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    perturbed_model = perturbed_model.cuda()
    perturbed_model = torch.compile(perturbed_model)
    rms = math.sqrt(norm_squared / count)
    std = rms * 1e-6
    print(std)
    perturb(state_dict, std)
    perturbed_model.load_state_dict(state_dict)
    perturbed_model = DDP(perturbed_model, device_ids=[ddp_local_rank])

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # run validation batches
    model.eval()
    perturbed_model.eval()
    val_gen = val_loader.token_generator()
    total = 0.0
    for _ in range(val_steps):
        x_val, _ = next(val_gen)
        with (ctx, torch.no_grad()):
            total += torch.sum((model(x_val) - perturbed_model(x_val)) ** 2).item()

    jacob_norm_squared = total / (val_steps * args.batch_size * args.sequence_length * std ** 2)
    print(f"{math.sqrt(jacob_norm_squared)=}")

    # clean up nice
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
