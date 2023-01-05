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

from sklearn import model_selection
from tqdm import tqdm

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import gc
gc.enable()

from args import args
from dataset import Data_class
from model import SEN_Model
from train_eval import train_fn, eval_fn

from loss import loss_fn

from fastapi import FastAPI

app = FastAPI()

class Data_class(Dataset):
    def __init__(self, df,args, inference_only=False):
        super().__init__()
        
        self.df = df      
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
        

class SEN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(args.BERT_PATH)
        config.update({"output_hidden_states":True, 
                       "layer_norm_eps": 1e-7})                       
        self.layer_start = 9
        self.bert = AutoModel.from_pretrained(args.BERT_PATH, config=config)  

        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.linear = nn.Linear(768, 1)
#         self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        #print(outputs)
        #all_hidden_state = outputs.hidden_states[-1]
       # weighted_pooling_embeddings = self.pooler(all_hidden_state)
#         print(outputs.hidden_states[-1].shape)
        
        weights = self.attention(outputs.hidden_states[-1])
        #[batch_size, max_len, hidden_states]
#         print(weights.shape)
        
       
        context_vector = torch.sum(weights *outputs.hidden_states[-1] , dim=1) 
#         print((weights *outputs.hidden_states[-1]).shape)
#         print(context_vector.shape)
        
        return self.linear(context_vector)
    
    
@app.get("/")
def read_root():
    return { "hello" : "World"}


# model_1 = SEN_Model()
model_2 = SEN_Model()
model_3 = SEN_Model()
# model_4 = SEN_Model()
# model_5 = SEN_Model()

# model_path_1 = r"\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights\model_{fold}_.pth".format(fold = 0)
# model_path_2 = r"\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights\model_{fold}_.pth".format(fold = 1)
# model_path_3 = r"\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights\model_{fold}_.pth".format(fold = 2)
# model_path_4 = r"\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights\model_{fold}_.pth".format(fold = 3)
# model_path_5 = r"\Users\ajayp\OneDrive\Desktop\Project\Saved_model_weights\model_{fold}_.pth".format(fold = 4)

model_path_2 = args.model_path + '\\' + r"model_{fold}_.pth".format(fold = 0)
model_path_3 = args.model_path + '\\' + r"model_{fold}_.pth".format(fold = 1)

# model_1.load_state_dict(torch.load(model_path_1))
# model_1.eval().cuda()

model_2.load_state_dict(torch.load(model_path_2))
model_2.eval().cuda()

model_3.load_state_dict(torch.load(model_path_3))
model_3.eval().cuda()

# model_4.load_state_dict(torch.load(model_path_4))
# model_4.eval().cuda()

# model_5.load_state_dict(torch.load(model_path_5))
# model_5.eval().cuda()


@app.get("/predict")
def fetch_predictions(text : str):
    data = {'Unnamed: 0': [0],
            'text': [text]}
    df = pd.DataFrame(data)

    temp = Data_class(df, args, inference_only = True)
    ans = temp.__getitem__(0)
    input_ids, attention_mask = ans["input_ids"], ans["attention_mask"]
    input_ids = input_ids.unsqueeze(0).cuda()
    attention_mask = attention_mask.unsqueeze(0).cuda()
    device = torch.device("cuda")
    
#     out_1 = model_1(input_ids, attention_mask)
    out_2 = model_2(input_ids, attention_mask)
    out_3 = model_3(input_ids, attention_mask)
#     out_4 = model_4(input_ids, attention_mask)
#     out_5 = model_5(input_ids, attention_mask)

    out_2 = torch.sigmoid(out_2).item()
    out_3 = torch.sigmoid(out_3).item()
    
    predictions = (out_2 + out_3)/2
    
#     predictions = torch.sigmoid(final_out).item()
   

    if predictions >= 0.75:
        sentiment = "Negative"
    else:
        sentiment = "Positive"
        
     
    return { "positive" : 1-predictions, "negative": predictions, "sentiment" : sentiment, "text" : text}
    

