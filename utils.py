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

from transformers import BertTokenizer, BertForSequenceClassification

def text2feature(movie_df,feature_fn=None,tokenizer=None):
    if feature_fn is None:
        return 
        
    """
    for loop for data frame... 
    """
    feature_embed = {}
    
    for item_id, movie_plot in : ....
    
        if tokenizer is not None:
            inputs = tokenizer(plot_synopsis, return_tensors="pt")
        else:
            """
            another mechanism to tokenize raw string 
            """
        features = model.forward(**inputs, labels=labels,output_hidden_states =True,return_dict=True).hidden_states[-1].squeeze()[0].tolist()
        
        feature_embed[item_id] = features
        
    return feature_embed


def bert2csv(bert_embedding,bert_dim,bert_filename='bert_sequence.txt'):
    N = bert_dim
    with open(bert_filename,'w') as f:
        for item_id,embedding in bert_embedding.items():
            f.write("{}".format(item_id)+(" {}"*N).format(*embedding) + "\n")
            
def bert2dict(bert_filename=r"bert_sequence.txt"):
    with open(bert_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        feature_embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
        
    return feature_embed

"""
add function to read movie df
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    
    parser.add_argument('--read_filename',type=str,help='The filename to read all the pre-computed feature embeddings from or to',default="bert_sequence.txt")
    parser.add_argument('--read', action='store_true',
                        help='--read if you want to read in the embeddings, otherwise, write the embeddings')

    read_filename = args.read_filename
    read = args.read
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    bert_dim = model.config.hidden_size
    
    if read:
        feature_embed = text2feature(movie_df,feature_fn=model,tokenizer=tokenizer)
        bert2csv(bert_embedding,bert_dim=bert_dim,bert_filename=read_filename)
    
    else:
        feature_embed = bert2dict(bert_filename=read_filename)
    