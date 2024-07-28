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



app = typer.Typer(
    
)

def train_step() - float: