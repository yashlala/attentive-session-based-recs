# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:41 2021

@author: lpott
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import torch

class DilatedResBlock(nn.Module):
    def __init__(self,dilation,channel,max_len):
        super(DilatedResBlock,self).__init__()
        self.dilation = dilation
        self.channel = channel
        self.half_channel = int(channel/2)
        self.max_len = max_len
        
        self.reduce = nn.Conv1d(channel,self.half_channel,1)
        self.masked = nn.Conv1d(self.half_channel,self.half_channel,3,dilation=dilation)
        self.increase = nn.Conv1d(self.half_channel,channel,1)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=[max_len])#channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        
    def forward(self,x):
        y = self.reduce_norm(x.permute(0,2,1)).permute(0,2,1)
        #y = self.reduce_norm(x)

        y = F.leaky_relu(x)
        y = self.reduce(y)
        
                
        y = self.masked_norm(y.permute(0,2,1)).permute(0,2,1)
        y = F.leaky_relu(y)
        y = F.pad(y,pad=(2 + (self.dilation-1)*2,0),mode='constant')
        y = self.masked(y)
      
        
        y = self.increase_norm(y.permute(0,2,1)).permute(0,2,1)
        #y = self.increase_norm(y)
        y = F.leaky_relu(y)
        y = self.increase(y)
        
        return x+y
        

class NextItNet(nn.Module):
    """

    """
    def __init__(self,embedding_dim,output_dim,max_len,hidden_layers=2,dilations=[1,2,4,8],pad_token=0):
        super(NextItNet,self).__init__()
        self.embedding_dim = embedding_dim
        self.channel = embedding_dim
        self.output_dim = output_dim
        self.pad_token = pad_token
        self.max_len = max_len
    
    
        self.item_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        self.hidden_layers = nn.Sequential(*[nn.Sequential(*[DilatedResBlock(d,embedding_dim,max_len) for d in dilations]) for _ in range(hidden_layers)])

        self.final_layer = nn.Linear(embedding_dim, output_dim)

    
    def forward(self,x):
        x = self.item_embedding(x).permute(0,2,1)
        x = self.hidden_layers(x)
        x = self.final_layer(x.permute(0,2,1))
        
        return x
    
class gru4recF(nn.Module):
    """
    embedding dim: the dimension of the item-embedding look-up table
    hidden_dim: the dimension of the hidden state of the GRU-RNN
    batch_first: whether the batch dimension should be the first dimension of input to GRU-RNN
    output_dim: the output dimension of the last fully connected layer
    max_length: the maximum session length for any user, used for packing/padding input to GRU-RNN
    pad_token: the value that pad tokens should be set to for GRU-RNN and item embedding
    bert_dim: the dimension of the feature-embedding look-up table
    ... to do add all comments ... 
    """
    def __init__(self,embedding_dim,hidden_dim,output_dim,genre_dim=0,batch_first=True,max_length=200,pad_token=0,pad_genre_token=0,bert_dim=0):
        super(gru4recF,self).__init__()
        
        self.batch_first =batch_first
        
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim
        self.genre_dim = genre_dim
        self.bert_dim = bert_dim

        self.max_length = max_length
        self.pad_token = pad_token
        self.pad_genre_token = pad_genre_token
    
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token
        if bert_dim != 0:
            self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            self.plot_embedding.requires_grad_(requires_grad=False)
            
            # project plot embedding to same dimensionality as movie embedding
            self.plot_projection = nn.Linear(bert_dim,embedding_dim)
        
        if genre_dim != 0:
            self.genre_embedding = nn.Embedding(genre_dim+1,embedding_dim,padding_idx=pad_genre_token)


        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first)

        # add 1 to the output dimension because we have to add a pad token
        self.output_layer = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        if (self.bert_dim != 0) and (self.genre_dim != 0):
            x = self.movie_embedding(x) + self.plot_projection(self.plot_embedding(x)) + self.genre_embedding(x_genre).sum(2)
        elif (self.bert_dim != 0) and (self.genre_dim == 0):
            x = self.movie_embedding(x) + self.plot_projection(self.plot_embedding(x)) 
        elif (self.bert_dim == 0) and (self.genre_dim != 0):
            x = self.movie_embedding(x) + self.genre_embedding(x_genre).sum(2)
        else:
            x = self.movie_embedding(x)
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
                
        return x
    
    def init_weight(self,reset_object,feature_embed):
        for (item_id,embedding) in feature_embed.items():
            if item_id not in reset_object.item_enc.classes_:
                continue
            item_id = reset_object.item_enc.transform([item_id]).item()
            self.plot_embedding.weight.data[item_id,:] = torch.DoubleTensor(embedding)
            
class gru4recFC(nn.Module):
    """
    embedding dim: the dimension of the item-embedding look-up table
    hidden_dim: the dimension of the hidden state of the GRU-RNN
    batch_first: whether the batch dimension should be the first dimension of input to GRU-RNN
    output_dim: the output dimension of the last fully connected layer
    max_length: the maximum session length for any user, used for packing/padding input to GRU-RNN
    pad_token: the value that pad tokens should be set to for GRU-RNN and item embedding
    bert_dim: the dimension of the feature-embedding look-up table
    ... to do add all comments ... 
    """
    def __init__(self,embedding_dim,hidden_dim,output_dim,genre_dim=0,batch_first=True,max_length=200,pad_token=0,pad_genre_token=0,bert_dim=0):
        super(gru4recFC,self).__init__()
        
        self.batch_first =batch_first
        
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim
        self.genre_dim = genre_dim
        self.bert_dim = bert_dim

        self.max_length = max_length
        self.pad_token = pad_token
        self.pad_genre_token = pad_genre_token
    
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token
        if bert_dim != 0:
            self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            self.plot_embedding.requires_grad_(requires_grad=False)
        
        if genre_dim != 0:
            self.genre_embedding = nn.Embedding(genre_dim+1,embedding_dim,padding_idx=pad_genre_token)

        self.projection_layer = nn.Linear(bert_dim+embedding_dim+genre_dim,embedding_dim)
        
        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first)

        # add 1 to the output dimension because we have to add a pad token
        self.output_layer = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        if (self.bert_dim != 0) and (self.genre_dim != 0):
            x = torch.cat( (self.movie_embedding(x),self.plot_embedding(x),self.genre_embedding(x_genre).sum(2)) , 2)
        elif (self.bert_dim != 0) and (self.genre_dim == 0):
            x = torch.cat( (self.movie_embedding(x),self.plot_embedding(x)) , 2)
        elif (self.bert_dim == 0) and (self.genre_dim != 0):
            x = torch.cat( (self.movie_embedding(x),self.genre_embedding(x_genre).sum(2)) , 2)
        else:
            x = self.movie_embedding(x)
        
        x = self.projection_layer(x)
        x = F.leaky_relu(x)
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
                
        return x
    
    def init_weight(self,reset_object,feature_embed):
        for (item_id,embedding) in feature_embed.items():
            if item_id not in reset_object.item_enc.classes_:
                continue
            item_id = reset_object.item_enc.transform([item_id]).item()
            self.plot_embedding.weight.data[item_id,:] = torch.DoubleTensor(embedding)
            
            
class gru4rec_vanilla(nn.Module):
    """
    ... to do add all comments ... 
    """
    def __init__(self,hidden_dim,output_dim,batch_first=True,max_length=200,pad_token=0):
        super(gru4rec_vanilla,self).__init__()
        
        self.batch_first =batch_first
        
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim

        self.max_length = max_length
        self.pad_token = pad_token
        
        self.genre_dim = 0
        self.bert_dim = 0
    
        # initialize item-id lookup table as one hot vector
        self.movie_embedding = torch.eye(output_dim+1).cuda()
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token

        self.encoder_layer = nn.GRU(output_dim+1,self.hidden_dim,batch_first=self.batch_first)

        # add 1 to the output dimension because we have to add a pad token
        self.output_layer = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...

        x = self.movie_embedding[x]
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
                
        return x
    
class gru4rec_feature(nn.Module):
    """
    embedding dim: the dimension of the item-embedding look-up table
    hidden_dim: the dimension of the hidden state of the GRU-RNN
    batch_first: whether the batch dimension should be the first dimension of input to GRU-RNN
    output_dim: the output dimension of the last fully connected layer
    max_length: the maximum session length for any user, used for packing/padding input to GRU-RNN
    pad_token: the value that pad tokens should be set to for GRU-RNN and item embedding
    bert_dim: the dimension of the feature-embedding look-up table
    ... to do add all comments ... 
    """
    def __init__(self,hidden_dim,output_dim,batch_first=True,max_length=200,pad_token=0,bert_dim=0):
        super(gru4rec_feature,self).__init__()
        
        self.batch_first =batch_first
        
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim

        self.max_length = max_length
        self.pad_token = pad_token
        
        self.genre_dim = 0
        self.bert_dim = 0
    
        # initialize item-id lookup table as one hot vector
        self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
        self.plot_embedding.requires_grad_(requires_grad=False)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token

        self.encoder_layer = nn.GRU(bert_dim,self.hidden_dim,batch_first=self.batch_first)

        # add 1 to the output dimension because we have to add a pad token
        self.output_layer = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...

        x = self.plot_embedding(x)
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
                
        return x
    
    def init_weight(self,reset_object,feature_embed):
        for (item_id,embedding) in feature_embed.items():
            if item_id not in reset_object.item_enc.classes_:
                continue
            item_id = reset_object.item_enc.transform([item_id]).item()
            self.plot_embedding.weight.data[item_id,:] = torch.DoubleTensor(embedding)