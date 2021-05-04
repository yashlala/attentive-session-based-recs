# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import csv 
import torch

from transformers import BertTokenizer, BertForSequenceClassification
from preprocessing import create_movie_df

def text2feature(movie_df,feature_fn=None,tokenizer=None,device=None):
    print("="*10,"Creating Text to Feature Dictionary","="*10)
    if feature_fn is None:
        return 
        
    """
    for loop for data frame... 
    """
    feature_embed = {}
    
    for (item_id,item_plot) in tqdm(movie_df.loc[:,['item_id','mplot']].values):
    
        inputs = tokenizer(item_plot, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
      
            
        features = model.forward(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,output_hidden_states=True,return_dict=True).hidden_states[-1].squeeze()[0].tolist()
        
        feature_embed[item_id] = features
        
    return feature_embed


def bert2csv(bert_embedding,bert_dim,bert_filename='bert_sequence.txt'):
    print("="*10,"Creating .txt file with all item id and embeddings","="*10)

    N = bert_dim
    with open(bert_filename,'w') as f:
        for (item_id,embedding) in tqdm(bert_embedding.items()):
            f.write("{}".format(item_id)+(" {}"*N).format(*embedding) + "\n")
            
def bert2dict(bert_filename=r"bert_sequence.txt"):
    print("="*10,"Reading .txt file with all item id and embeddings","="*10)
    with open(bert_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        feature_embed = {float(line[0]): np.array(list(map(float, line[1:])))
                for line in reader}
        
    return feature_embed

"""
add function to read movie df
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--read_bert_filename',type=str,help='The filename to read all the pre-computed feature embeddings from or to',default="bert_sequence.txt")
    parser.add_argument('--read', action='store_true',
                        help='--read if you want to read in the embeddings, otherwise, write the embeddings')
    parser.add_argument('--read_movie_filename',type=str,help='The filename to read all the movie information from',default="movies-20m.csv")
    parser.add_argument('--size',type=str,help='The dataset (1m , 20m , etc) which you will use',default="20m")

    args = parser.parse_args()

    read_bert_filename = args.read_bert_filename
    read_movie_filename = args.read_movie_filename
    size=args.size

    read = args.read
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',truncation=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    bert_dim = model.config.hidden_size
    
    movie_df = create_movie_df(read_movie_filename,size=size)
    
    if not read:
        print("Writing")
        bert_embedding = text2feature(movie_df,feature_fn=model,tokenizer=tokenizer,device=device)
        bert2csv(bert_embedding,bert_dim=bert_dim,bert_filename=read_bert_filename)
    
    else:
        feature_embed = bert2dict(bert_filename=read_bert_filename)
    