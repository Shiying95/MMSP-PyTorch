import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from mmsp.modules.dist_utils import all_gather
from mmsp.gadgets.my_metrics import (
    Accuracy, Scalar, 
    Wmape, Wmape_region_dt, Wmape_region, Wmape_all, Mask_mse_loss, Acc, MAE
    )


# 计算metrics
def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == 'sp':
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                metric_funcs = {
                    'wmape': Wmape(),
                    'wmape_region_dt': Wmape_region_dt(),
                    'wmape_region': Wmape_region(),
                    'wmape_all': Wmape_all(),
                    'acc': Acc(),
                    'mae': MAE(),
                }
                for metric in pl_module.metrics:
                    setattr(pl_module, f"{split}_{k}_{metric}", metric_funcs[metric])
                
            elif k == 'tit':
                for i, col in enumerate(pl_module.tit_cols):
                    name = f'{col}_classifier'
                    setattr(pl_module, f"{split}_{name}_accuracy", Accuracy())
                    setattr(pl_module, f"{split}_{name}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"

    pl_module.print("")  # 强制换行
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        if loss_name == "tit":
            for i, col in enumerate(pl_module.tit_cols):
                name = f'{col}_classifier'

                for metric in ['loss', 'accuracy']:
                    log_value = getattr(pl_module, f"{phase}_{name}_{metric}").compute()
                    getattr(pl_module, f"{phase}_{name}_{metric}").reset()
                    log_name = f"epoch_{loss_name}/{phase}/{name}_{metric}"
                    pl_module.log(log_name, log_value)
                    pl_module.print(f"{log_name}: {log_value:.4f}")

                    if (metric == pl_module.hp_metric) and (phase != 'train'):
                        pl_module.log("hp_metric", log_value)  # show in the tensorboard/hparams           

        elif loss_name == "sp":
            for metric in ['loss'] + pl_module.metrics:
                log_value = getattr(pl_module, f"{phase}_{loss_name}_{metric}").compute()
                getattr(pl_module, f"{phase}_{loss_name}_{metric}").reset()
                log_name = f"epoch_{loss_name}/{phase}/{metric}"
                pl_module.log(log_name, log_value)
                pl_module.print(f"{log_name}: {log_value:.4f}")

                if (metric == pl_module.hp_metric) and (phase != 'train'):
                    pl_module.log("hp_metric", log_value)  # show in the tensorboard/hparams           

        else:
            for metric in ['loss', 'accuracy']:
                log_value = getattr(pl_module, f"{phase}_{loss_name}_{metric}").compute()
                getattr(pl_module, f"{phase}_{loss_name}_{metric}").reset()
                log_name = f"epoch_{loss_name}/{phase}/{loss_name}_{metric}"
                pl_module.log(log_name, log_value)
                pl_module.print(f"{log_name}: {log_value:.4f}")


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    elif optim_type == "Adam":
        optimizer = torch.optim.Adam(pl_module.parameters(), lr=lr, weight_decay=wd)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
