import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src.config import STOPWORDS

def load_datas(dataset_loc: str, num_samples: int = None) -> Dataset:
    """Load data from source into a Ray Dataset

    Args:
        dataset_loc (str): Location of the dataset
        num_samples (int, optional): The number of samples to load. Default to None

    Returns:
        Dataset: _description_
    """
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): _description_
        stratify (str): _description_
        test_size (float): _description_
        shuffle (bool, optional): _description_. Defaults to True.
        seed (int, optional): _description_. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets
    """
    def _add_split(df: pd.DataFrame) -> pd.DataFrame:
        """Naively split a dataframe into train and test splits.
        Add  a column specifying whether it's the tain or test slit.
        """
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])
    
    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[df["_split"] == split].drop("_split", axis=1)
    
    
    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")
    
    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)
    
    return train_ds, test_ds


def clean_text(text, stopwords=STOPWORDS):
    """Tokenize the  text input our bath using a tokenizer

    Args:
        text (_type_): _description_
        stopwords (_type_, optional): _description_. Defaults to STOPWORDS.

    Returns:
        _type_: _description_
    """
    # Lower
    text = text.lower()
    
    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)
    
    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) #remove non alphanumber
    text = re.sub(" +", " ", text) # remove multiple spaces
    text = text.strip()
    text = re.sub(r"http\S+", "", text) # remove liks
    
    return text

def tokenize(batch: Dict) -> Dict:
    """_summary_

    Args:
        batch (Dict): _description_

    Returns:
        Dict: _description_
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_input = tokenizer(batch["text"].tolist(), return_tensor="np", padding="longest") 
    return dict(ids=encoded_input["inputs_ids"], masks=encoded_input["attention_mask"], targets=np.array(batch["tag"])) 


def preprocess(df, class_to_index):
    """_summary_

    Args:
        df (_type_): _description_
        class_to_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    df["text"] = df.title + " " + df.description # featuring engineering
    df["text"] = df.text.apply(clean_text)
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore" )
    df = df[["text", "tag"]] # rearrange columns
    df["tag"] = df["tag"].map(class_to_index) # label encoding
    outputs = tokenize(df)
    return outputs  

class CustomPreprocessor:
    """Custom preprocessor class.
    """
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {} #multi defaluts
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        
    def fit(self, ds):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self
    
    def transform(self, ds):
        return ds.map_batches(preprocess, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")    

    