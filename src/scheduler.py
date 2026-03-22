# First half Taken from https://github.com/ipis-mjkim/caueeg-ceednet and extensively modified

import logging
import math
import time
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

log = logging.getLogger(__name__)

lr_scheduler_list = [
    "constant_with_decay",
    "constant_with_twice_decay",
    "transformer_style",
    "cosine_decay_with_warmup_half",
    "cosine_decay_with_warmup_one_and_half",
    "cosine_decay_with_warmup_two_and_half",
    "linear_decay_with_warmup",
    "cosine_annealing"
]


def get_constant_with_decay_scheduler(optimizer: Optimizer, iterations: int, last_epoch: int = -1):
    return lr_scheduler.StepLR(optimizer, step_size=round(iterations * 0.8), gamma=0.1, last_epoch=last_epoch)


def get_constant_with_twice_decay_scheduler(optimizer: Optimizer, iterations: int, last_epoch: int = -1):
    return lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(iterations * 0.7), round(iterations * 0.9)], gamma=0.1, last_epoch=last_epoch
    )


def get_transformer_style_scheduler(optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
    def transformer_style_lambda(step: int):
        return min(math.sqrt(warmup_steps) / max(1.0, math.sqrt(step)), step / max(1.0, float(warmup_steps)))

    return lr_scheduler.LambdaLR(optimizer, transformer_style_lambda, last_epoch=last_epoch)


def get_cosine_decay_with_warmup(
    optimizer: Optimizer, warmup_steps: int, iterations: int, cycles: float = 0.5, last_epoch: int = -1
):
    def cosine_decay_with_warmup_lambda(step):
        if step <= warmup_steps:
            return step / max(1.0, float(warmup_steps))
        period = (step - warmup_steps) / max(1.0, float(iterations - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * period)))

    return lr_scheduler.LambdaLR(optimizer, cosine_decay_with_warmup_lambda, last_epoch)


def get_linear_decay_with_warmup(optimizer: Optimizer, warmup_steps: int, iterations: int, last_epoch: int = -1):
    def linear_decay_with_warmup(step: int):
        if step <= warmup_steps:
            return step / max(1.0, float(warmup_steps))
        return max(0.0, (iterations - step) / max(1.0, float(iterations - warmup_steps)))

    return lr_scheduler.LambdaLR(optimizer, linear_decay_with_warmup, last_epoch)

def get_lr_scheduler(
    optimizer: Optimizer, scheduler_type: str, iterations: int, warmup_steps: int, last_epoch: int = -1, min_lr: float = 0
):
    if scheduler_type == "constant_with_decay":
        return get_constant_with_decay_scheduler(optimizer=optimizer, iterations=iterations, last_epoch=last_epoch)
    elif scheduler_type == "constant_with_twice_decay":
        return get_constant_with_twice_decay_scheduler(
            optimizer=optimizer, iterations=iterations, last_epoch=last_epoch
        )
    elif scheduler_type == "transformer_style":
        return get_transformer_style_scheduler(optimizer=optimizer, warmup_steps=warmup_steps, last_epoch=last_epoch)
    elif scheduler_type == "cosine_decay_with_warmup_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=0.5, last_epoch=last_epoch
        )
    elif scheduler_type == "cosine_decay_with_warmup_one_and_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=1.5, last_epoch=last_epoch
        )
    elif scheduler_type == "cosine_decay_with_warmup_two_and_half":
        return get_cosine_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, cycles=2.5, last_epoch=last_epoch
        )
    elif scheduler_type == "linear_decay_with_warmup":
        return get_linear_decay_with_warmup(
            optimizer=optimizer, warmup_steps=warmup_steps, iterations=iterations, last_epoch=last_epoch
        )
    else:
        raise ValueError(
            f"ERROR: get_lr_scheduler(scheduler_type) input is not understandable: {scheduler_type}. "
            f"Check the input value again: {lr_scheduler_list}"
        )


##########

def get_timm_cosine_decay(cfg, optimizer):

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=cfg.model.t_initial, # The number of epochs in one cosine period (before lr resets)
        lr_min=cfg.model.min_lr,
        warmup_t=cfg.model.warmup_t,
        warmup_lr_init=cfg.model.warmup_lr_init,
        t_in_epochs=True,
        warmup_prefix=cfg.model.warmup_prefix,
        cycle_limit=1000000000,
        cycle_decay=cfg.model.decay_rate
    )

    return scheduler

def get_pytorch_cosine_decay():
    pass

def get_multistep_lr(cfg, optimizer):
    scheduler = lr_scheduler.MultiStepLR(
        milestones=list(cfg.model.milestones),
        gamma=cfg.model.gamma
    )
    return scheduler

def make_scheduler(cfg, optimizer):

    scheduler = None
    scheduler_name = cfg.model.scheduler
    if scheduler_name == "timm-cosine-decay":
        scheduler = get_timm_cosine_decay(cfg, optimizer)
    elif scheduler_name == "multistep-lr":
        scheduler = get_multistep_lr(cfg, optimizer)
    elif scheduler_name == "cosine_decay_with_warmup_half":
        iterations = (cfg.model.epochs * cfg.train.samples_per_epoch) // cfg.model.minibatch
        warmup_steps = (cfg.model.warmup_t/100) * iterations

        scheduler = get_lr_scheduler(optimizer, scheduler_name, iterations, warmup_steps)
    else:
        log.warning("No scheduler selected! Please verify that this is the intended behavior.")
        time.sleep(3)

    return scheduler