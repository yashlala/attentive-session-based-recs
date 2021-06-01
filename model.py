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
    def __init__(self,embedding_dim,
                 output_dim,
                 max_len,
                 hidden_layers=2,
                 dilations=[1,2,4,8],
                 pad_token=0):
        
        super(NextItNet,self).__init__()
        self.embedding_dim = embedding_dim
        self.channel = embedding_dim
        self.output_dim = output_dim
        self.pad_token = pad_token
        self.max_len = max_len
    
        self.genre_dim = 0
        self.bert_dim = 0
    
        self.item_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        self.hidden_layers = nn.Sequential(*[nn.Sequential(*[DilatedResBlock(d,embedding_dim,max_len) for d in dilations]) for _ in range(hidden_layers)])

        self.final_layer = nn.Linear(embedding_dim, output_dim)

    
    def forward(self,x,x_lens=None):
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
    def __init__(self,embedding_dim,
                 hidden_dim,
                 output_dim,
                 genre_dim=0,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 pad_genre_token=0,
                 bert_dim=0,
                 dropout=0,
                 tied=False):
        
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
        
        self.tied = tied
        self.dropout = dropout
        
        if self.tied:
            self.hidden_dim = embedding_dim
    
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token
        if bert_dim != 0:
            self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding.requires_grad_(requires_grad=False)
            #self.plot_embedding = torch.ones(output_dim+1,bert_dim).cuda() #nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding[pad_token,:] = 0
            
            # project plot embedding to same dimensionality as movie embedding
            self.plot_projection = nn.Linear(bert_dim,embedding_dim)
                    
        if genre_dim != 0:
            self.genre_embedding = nn.Embedding(genre_dim+1,embedding_dim,padding_idx=pad_genre_token)


        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first,dropout=self.dropout)

        # add 1 to the output dimension because we have to add a pad token
        if not self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim)
        
        if self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim+1)
            self.output_layer.weight = self.movie_embedding.weight
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        if (self.bert_dim != 0) and (self.genre_dim != 0):
            x = self.movie_embedding(x) + self.plot_projection(F.leaky_relu(self.plot_embedding(x))) + self.genre_embedding(x_genre).sum(2)
        elif (self.bert_dim != 0) and (self.genre_dim == 0):
            x = self.movie_embedding(x) + self.plot_projection(F.leaky_relu(self.plot_embedding(x)))
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
    def __init__(self,embedding_dim,
                 hidden_dim,
                 output_dim,
                 genre_dim=0,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 pad_genre_token=0,
                 bert_dim=0,
                 tied=False,dropout=0):
        
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
        self.tied = tied
        
        self.dropout = dropout
        if self.tied:
            self.hidden_dim = embedding_dim
            
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token
        if bert_dim != 0:
            self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding.requires_grad_(requires_grad=False)
            #self.plot_embedding = torch.ones(output_dim+1,bert_dim).cuda() #nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding[pad_token,:] = 0
        
        if genre_dim != 0:
            self.genre_embedding = nn.Embedding(genre_dim+1,embedding_dim,padding_idx=pad_genre_token)
            self.projection_layer = nn.Linear(bert_dim+embedding_dim+embedding_dim,embedding_dim)
        
        else:
            self.projection_layer = nn.Linear(bert_dim+embedding_dim,embedding_dim)
        
        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first,dropout=self.dropout)

        # add 1 to the output dimension because we have to add a pad token
        if not self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim)
        
        if self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim+1)
            self.output_layer.weight = self.movie_embedding.weight
            
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        if (self.bert_dim != 0) and (self.genre_dim != 0):
            x = torch.cat( (self.movie_embedding(x),self.plot_embedding(x),self.genre_embedding(x_genre).sum(2)) , 2)
        elif (self.bert_dim != 0) and (self.genre_dim == 0):
            x = torch.cat( (self.movie_embedding(x),self.plot_embedding(x) ) , 2)
        elif (self.bert_dim == 0) and (self.genre_dim != 0):
            x = torch.cat( (self.movie_embedding(x),self.genre_embedding(x_genre).sum(2)) , 2)
        else:
            x = self.movie_embedding(x)
        
        x = F.leaky_relu(x)
        
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
            
class gru4rec_conv(nn.Module):
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
    def __init__(self,embedding_dim,
                 hidden_dim,
                 output_dim,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 dropout=0,
                 window=3,
                 tied=False):
        
        super(gru4rec_conv,self).__init__()
        
        self.batch_first =batch_first
        
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim
        self.window = window
        self.conv_embed = int(embedding_dim//2)

        self.max_length = max_length
        self.pad_token = pad_token
        
        self.tied = tied
        self.dropout = dropout
        
        self.genre_dim = 0
        self.bert_dim = 0
        
        if self.tied:
            self.hidden_dim = embedding_dim
    
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
            
        # project plot embedding to same dimensionality as movie embedding
        self.projection = nn.Conv1d(self.embedding_dim,self.conv_embed,self.window)

        self.encoder_layer = nn.GRU(self.conv_embed,self.hidden_dim,batch_first=self.batch_first,dropout=self.dropout)

        # add 1 to the output dimension because we have to add a pad token
        if not self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim)
        
        if self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim+1)
            self.output_layer.weight = self.movie_embedding.weight
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        
        x = self.movie_embedding(x).permute(0,2,1)
        x = F.pad(x,pad=(self.window-1,0),mode='constant')
        x = self.projection(x).permute(0,2,1)
        x = F.leaky_relu(x)
                    
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        output_packed,_ = self.encoder_layer(x)        
        
        if pack:
            x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        x = self.output_layer(x)
        
                
        return x
            
class gru4rec_vanilla(nn.Module):
    """
    ... to do add all comments ... 
    """
    def __init__(self,hidden_dim,
                 output_dim,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 tied=False,
                 embedding_dim=0,
                 device='cpu'):
        
        super(gru4rec_vanilla,self).__init__()
        
        self.batch_first =batch_first
        
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim
        self.embedding_dim = embedding_dim

        self.max_length = max_length
        self.pad_token = pad_token
        
        self.genre_dim = 0
        self.bert_dim = 0
        
        self.tied = tied
        self.embedding_dim = embedding_dim
    
        if self.tied:
            self.hidden_dim = embedding_dim
    
        # initialize item-id lookup table as one hot vector
        if self.embedding_dim == 0:
            self.movie_embedding = torch.eye(output_dim+1).to(device)
            
        elif self.embedding_dim != 0:
            self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token

        if self.embedding_dim == 0:
            self.encoder_layer = nn.GRU(output_dim+1,self.hidden_dim,batch_first=self.batch_first)
            
        elif self.embedding_dim != 0:
            self.encoder_layer = nn.GRU(self.embedding_dim,self.hidden_dim,batch_first=self.batch_first)


        # add 1 to the output dimension because we have to add a pad token
        if not self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim)
        
        if self.tied:
            self.output_layer = nn.Linear(hidden_dim,output_dim+1)
            self.output_layer.weight = self.movie_embedding.weight
    
    def forward(self,x,x_lens,x_genre=None,pack=True):
        # add the plot embedding and movie embedding
        # do I add non-linearity or not? ... 
        # concatenate or not? ...
        # many questions ...
        
        if self.embedding_dim == 0:
            x = self.movie_embedding[x]
            
        elif self.embedding_dim != 0:
            x = self.movie_embedding(x)
                    
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
    def __init__(self,hidden_dim,
                 output_dim,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 bert_dim=0):
        
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
        #self.plot_embedding = torch.ones(output_dim+1,bert_dim).cuda() #nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
        #self.plot_embedding[pad_token,:] = 0
        #self.plot_embedding.requires_grad_(requires_grad=False)
        
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



# This code was originally an implementation of Li, Jing, et al.
# It originated from:
# https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model.

    Note that this module performs its own embedding on the input.
    This may be unnecessary, as we're already getting some high quality BERT
    embeddings. TODO: Check this with Michael and Hamlin.

    Args:
        n_items(int):       The number of items we'll be receiving as input
                            (ie. the vocabulary size).
        hidden_size(int):   The width of a GRU hidden layer.
        embedding_dim(int): The dimension of our item embedding.
        batch_size(int):    The batch size for our network training.
        n_layers(int):      The number of GRU layers to use.
    """
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        # TODO: we embed the input again here. Is this necessary if we already
        # have our BERT module set up before this?
        # ANSWER: No. We don't need any embedding here.
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)

        # GRU + Attention layer
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)

        # Final feedforward fully-connected layer.
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        """Predict (unnormalized) item scores from an input sequence.

        Args:
            Seq: An input sequence of items to predict. These items should be
                 given in ItemID form -- embeddings are module-internal as of
                 now.
        """
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # Fetch the last hidden state of the last timestamp.
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size))
        alpha = alpha.view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))

        # Final fully-connected linear layer.
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size),
                requires_grad=True).to(self.device)
    

class gru4recF_attention(nn.Module):
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
    def __init__(self,embedding_dim,
                 hidden_dim,
                 output_dim,
                 attn_dim,
                 genre_dim=0,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 pad_genre_token=0,
                 bert_dim=0,
                 dropout=0,
                 tied=False,
                 cat=True,
                 attn=True):
        
        super(gru4recF_attention,self).__init__()
        
        self.batch_first =batch_first
        
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.output_dim =output_dim
        self.genre_dim = genre_dim
        self.bert_dim = bert_dim
        self.attn_dim = attn_dim

        self.max_length = max_length
        self.pad_token = pad_token
        self.pad_genre_token = pad_genre_token
        
        self.tied = tied
        self.dropout = dropout
        self.cat = cat
        self.attn = attn
        
        if self.tied:
            self.hidden_dim = embedding_dim
    
        # initialize item-id lookup table
        # add 1 to output dimension because we have to add a pad token
        self.movie_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        #  initialize plot lookup table
        # add 1 to output dimensino because we have to add a pad token
        if bert_dim != 0:
            self.plot_embedding = nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding.requires_grad_(requires_grad=False)
            #self.plot_embedding = torch.ones(output_dim+1,bert_dim).cuda() #nn.Embedding(output_dim+1,bert_dim,padding_idx=pad_token)
            #self.plot_embedding[pad_token,:] = 0
            
            # project plot embedding to same dimensionality as movie embedding
            self.plot_projection = nn.Linear(bert_dim,embedding_dim)
                    
        if genre_dim != 0:
            self.genre_embedding = nn.Embedding(genre_dim+1,embedding_dim,padding_idx=pad_genre_token)


        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first,dropout=self.dropout)
        
        if attn:
            self.attention_layer = nn.Linear(self.hidden_dim,self.attn_dim)
            self.score_layer = nn.Linear(self.attn_dim*2,1)
            self.sigmoid = nn.Sigmoid()

        if cat:
            attn_dim = attn_dim + hidden_dim # ht cat weightedSum
        
        # add 1 to the output dimension because we have to add a pad token
        if not self.tied:
            self.output_layer = nn.Linear(attn_dim,output_dim)
        
        if self.tied:
            self.output_layer = nn.Linear(attn_dim,output_dim+1)
            self.output_layer.weight = self.movie_embedding.weight
    
    def forward(self,x,x_lens,x_genre=None,pack=True,**kwargs):
        batch_size = x.size()[0]
        if (self.bert_dim != 0) and (self.genre_dim != 0):
            x = self.movie_embedding(x) + self.plot_projection(F.leaky_relu(self.plot_embedding(x))) + self.genre_embedding(x_genre).sum(2)
        elif (self.bert_dim != 0) and (self.genre_dim == 0):
            x = self.movie_embedding(x) + self.plot_projection(F.leaky_relu(self.plot_embedding(x)))
        elif (self.bert_dim == 0) and (self.genre_dim != 0):
            x = self.movie_embedding(x) + self.genre_embedding(x_genre).sum(2)
        else:
            x = self.movie_embedding(x)
        
        if pack:
            x = pack_padded_sequence(x,x_lens,batch_first=True,enforce_sorted=False)
        
        encoder_states, _ = self.encoder_layer(x) 
        if pack:
            encoder_states, _ = pad_packed_sequence(encoder_states, batch_first=self.batch_first,total_length=self.max_length,padding_value=self.pad_token)
            
        if self.attn:
            attn_states = self.attention_layer(encoder_states)
        
        # CCs = BS x MS x 2HS
        combined_contexts = torch.zeros(batch_size,self.max_length,self.attn_dim)
        if torch.cuda.is_available():
            combined_contexts = combined_contexts.cuda()
        
        for t in range(self.max_length):
            # CF = BS x (t+1) x HS
            context_frame = attn_states[:,:t+1,:]
            # CH = BS x HS x 1
            current_hidden = attn_states[:,t,:].squeeze(1).unsqueeze(2)
            # AS = BS x (t+1) x 1
            attention_score = torch.bmm(context_frame,current_hidden).squeeze(2) / self.attn_dim
            attention_score = torch.nn.functional.softmax(attention_score,1).unsqueeze(2)
            # CFT = BS x HS x (t+1)
            context_frame_transposed = torch.transpose(context_frame,1,2)
            # CV = BS x HS
            context_vector = torch.bmm(context_frame_transposed,attention_score).squeeze(2)
            # CH = BS x HS
            #current_hidden = current_hidden.squeeze(2)
            # CC = BS x AS
            combined_contexts[:,t,:] = context_vector
        

        if self.cat:
            ## CCs = BS x MS x AS, ES = BS x MS x HS
            combined_contexts_cat = torch.cat((combined_contexts,encoder_states),2)
            
        # CCC = BS x MS x (AS + HS)
        # O = BS x MS x V
        if self.cat:
            x = self.output_layer(combined_contexts_cat)
        else:
            x = self.output_layer(combined_contexts)
        return x
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def init_weight(self,reset_object,feature_embed):
        for (item_id,embedding) in feature_embed.items():
            if item_id not in reset_object.item_enc.classes_:
                continue
            item_id = reset_object.item_enc.transform([item_id]).item()
            self.plot_embedding.weight.data[item_id,:] = torch.DoubleTensor(embedding)
