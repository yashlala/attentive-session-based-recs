# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:41 2021

@author: lpott
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class gru4rec(nn.Module):
    """
    d_model - the number of expected features in the input
    nhead - the number of heads in the multiheadattention models
    dim_feedforward - the hidden dimension size of the feedforward network model
    """
    def __init__(self,embedding_dim,hidden_dim,output_dim,batch_first=True,max_length=200,pad_token=0):
        super(gru4rec,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.batch_first =batch_first
        self.output_dim =output_dim
        self.max_length = 200
        self.pad_token = pad_token
    
        
        self.move_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)

        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first)

        self.output_layer = nn.Linear(hidden_dim,output_dim+1)
    
    def forward(self,x,x_lens,pack=True):
        x = self.move_embedding(x)
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
                
        return x