import datetime
import os
import json
import tempfile
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer 
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DictributedDataParallel
from transformers import BertModel
from typing_extensions import Annotated


from src import data, utils


app = typer.Typer()

def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer
) -> float:
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size = batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad() # rest gradients
        z = model(batch) # forward pass
        targets = F.one_hot(batch["target"], num_classes=num_classes).float() # one-hot(for loss_fn)
        J = loss_fn(z, targets) # define loss
        J.backward() # backward pass
        optimizer.step() # update weights
        loss += (J.detach().item() - loss) / (i + 1) # commulative loss
    return loss

def eval_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer
) -> Tuple[float, np.array, np.array]: # program: no cover, tested via train workload
    model.train()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size = batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            optimizer.zero_grad() # rest gradients
            z = model(batch) # forward pass
            targets = F.one_hot(batch["target"], num_classes=num_classes).float() # one-hot(for loss_fn)
            J = loss_fn(z, targets).items()
            loss += (J.detach().item() - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.ectend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues).np.vstack(y_preds) 

def train_loop_per_worker(config: dict) -> None:
    


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    app()