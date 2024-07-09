import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from ray.train.torch import get_device

from src.config import mlflow

DatasetContent.get_current().execution_options.preserve_order = True

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(see)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'deterministic', False)")
    os.eviron["PYTHONHASHSEED"] = str(seed)
    
def load_dict(path: str) -> Dict:
    """Load a dictionary from a JSON'S filepath
    Args: 
        path (str): Location of file.
    
    Return:
        Dict: loaded JSON data       
    """
    with open(path) as fp:
        d = json.load(fp)
    return d    

def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location

    Args:
        d (Dict): data to save
        path (str): data of where to save the data
        cls (Any, optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    dictionary = os.path.dirname(path)
    if dictionary and not os.path.exists(dictionary): # progrma: no cover
        os.makedirs(dictionary)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fb, cls=cls, sort_keys=sortkeys)
        fp.write("\n")   
        
def pad_arry(arr: np.ndarray, dtype:np.int32) -> np.ndarray:
    """Pad an 2D array with seros until all raws in the
    2D array are of the same length as the logest
    row in the 2D array

    Args:
        arr (np.ndarray): input array
        dtype (np.int32): _description_

    Returns:
        np.ndarray: sero padded array
    """
    max_len = max(len(row) for row in arr) 
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr  


def collate_fn(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]: 
    """Convert a batch of numpy arrays to tensor (with appropriate padding)

    Args:
        batch (Dict[str, np.ndarray]): Input batch as a dictionary of numpy arrays

    Returns:
        Dict[str, torch.Tensor]: output batch as a dictionary of tensors.
    """
    batch["ids"] = pad_arry(batch["ids"])
    batch["mask"] = pad_arry(batch["mask"])
    dtypes = {"id": torch.int32, "mask": torch.int32, "target": torch.int64}
    tensor_batch = {}
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device) 
    return tensor_batch  

def get_run_id(experiment_name: str, trial_id: str) -> str:
    """Get the MLflow ru ID for  a specific Ray trial ID

    Args:
        experiment_name (str): name of the experiment
        trial_id (str): id of the trial

    Returns:
        str: run id of the trial
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.trial_naem = '{trial_name}").iloc[0]
    return run.run_id

def dic_to_list(data: Dict, keys: List[str]) -> List[Dict[str, any]]:
    """Convert a dictionary to a list of dictionaries

    Args:
        data (Dict): input dictionary
        keys (List[str]): key to include in the output list of dictionary

    Returns:
        List[Dict[str, any]]: output list of dictionry
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts     
    

    
    
 
    
    
    
    
    
    
    