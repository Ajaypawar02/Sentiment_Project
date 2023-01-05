
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from args import args, add_sentiment
from transformers import AdamW




class Data_class(Dataset):
    def __init__(self, df,args, inference_only=False):
        super().__init__()
        
        self.df = df      
        df["airline_sentiment"] = df["airline_sentiment"].apply(lambda x : add_sentiment(x))
        self.inference_only = inference_only
        self.text = df.text.tolist()
        
        if not self.inference_only:
            self.target = torch.tensor(df.airline_sentiment.values, dtype=torch.float)        
    
        self.encoded = args.tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = args.MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        
        if self.inference_only:
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
            }           
        else:
            target = self.target[index]
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
                "target" : target
            }

