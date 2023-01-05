import os
import math
import random
import time

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import BertModel, BertTokenizer

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import model_selection
from tqdm import tqdm

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import gc
gc.enable()



class args:
    train_path = r"C:\Users\ajayp\OneDrive\Desktop\Project\airline_sentiment_analysis.csv"
    TOKENIZER_PATH = "bert-base-uncased"
    BERT_PATH = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    MAX_LEN = 256
    train_batch_size = 1
    valid_batch_size = 1
    epochs = 2
    model_path = r"C:\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights"
    folds_path = r"C:\Users\ajayp\OneDrive\Desktop\Project\data\train_folds.csv"
    splits  = 20
    
    
def add_sentiment(row):
    if row == "positive":
        return 0
    else:
        return 1
