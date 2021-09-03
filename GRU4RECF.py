# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # GRU4REC-F
# 
# This notebook trains models from `models.py` on the input data, and evaluates their performance. 
# 
# It's set to train GRU4REC-F (our proposed non-attentive model) by default. 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import argparse
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from preprocessing import *
from dataset import *
from metrics import *
from model import *
from utils import bert2dict


# %%
# variables

read_filename ="data/movielens-20m/ratings.csv"
read_bert_filename = "data/bert_sequence_20m.txt"
read_movie_filename = ""#"movies-1m.csv"
size = "20m"

num_epochs = 50
lr =  0.0005
lr_alternate = 0.001
batch_size = 64
reg = 0#1e-6# was 1e-5 before
train_method = "alternate"
loss_type = "BPR_MAX"
num_neg_samples = 25
reg_bpr = 0


hidden_dim = 1024
embedding_dim = 1024
bert_dim= 768
window = 0

freeze_plot = False
tied = False
dropout= 0

k = 10
max_length = 200
min_len = 5


# nextitnet options...
hidden_layers = 3
dilations = [1,2,4,16]

model_type = "feature_add"

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


# %%
torch.cuda.empty_cache()


# %%
# ------------------Data Initialization----------------------#

# convert .dat file to time-sorted pandas dataframe
ml_1m = create_df(read_filename,size=size)

# remove users who have sessions lengths less than min_len
ml_1m = filter_df(ml_1m,item_min=min_len)


# %%
# ------------------Data Initialization----------------------#
if read_movie_filename != "":
    ml_movie_df = create_movie_df(read_movie_filename,size=size)
    ml_movie_df = convert_genres(ml_movie_df)
    
    # initialize reset object
    reset_object = reset_df()
    
    # map all user ids, item ids, and genres to range 0 - number of users/items/genres
    ml_1m,ml_movie_df = reset_object.fit_transform(ml_1m,ml_movie_df)
    
    # value that padded genre tokens shall take
    pad_genre_token = reset_object.genre_enc.transform(["NULL"]).item()
    
    genre_dim = len(np.unique(np.concatenate(ml_movie_df.genre))) - 1

else:
    # initialize reset object
    reset_object = reset_df()
    
    # map all user ids and item ids to range 0 - Number of Users/Items 
    # i.e. [1,7,5] -> [0,2,1]
    ml_1m = reset_object.fit_transform(ml_1m)
    
    pad_genre_token = None
    ml_movie_df = None
    genre_dim = 0


# %%
# ------------------Data Initialization----------------------#
# how many unique users, items, ratings and timestamps are there
n_users,n_items,n_ratings,n_timestamp = ml_1m.nunique()
print("Number of Users {:d}".format(n_users))
print("Number of Items {:d}".format(n_items))

# value that padded tokens shall take
pad_token = n_items

# the output dimension for softmax layer
output_dim = n_items


# get the item id : bert plot embedding dictionary
if bert_dim != 0:
    feature_embed = bert2dict(bert_filename=read_bert_filename)


# %%
# create a dictionary of every user's session (history)
# i.e. {user: [user clicks]}
if size == "1m":
    user_history = create_user_history(ml_1m)

elif size == "20m":
    # user_history = create_user_history(ml_1m)
    import pickle
    with open('userhistory.pickle', 'rb') as handle:
        user_history = pickle.load(handle)
    # with open('userhistory.pickle', 'wb') as handle:
    #     pickle.dump(user_history,handle, protocol=pickle.HIGHEST_PROTOCOL)
# create a dictionary of all items a user has not clicked
# i.e. {user: [items not clicked by user]}
# user_noclicks = create_user_noclick(user_history,ml_1m,n_items)


# %%
# split data by leave-one-out strategy
# have train dictionary {user: [last 41 items prior to last 2 items in user session]}
# have val dictionary {user: [last 41 items prior to last item in user session]}
# have test dictionary {user: [last 41 items]}
# i.e. if max_length = 4, [1,2,3,4,5,6] -> [1,2,3,4] , [2,3,4,5] , [3,4,5,6]
train_history,val_history,test_history = train_val_test_split(user_history,max_length=max_length)

# initialize the train,validation, and test pytorch dataset objects
# eval pads all items except last token to predict
train_dataset = GRUDataset(train_history,genre_df=ml_movie_df,mode='train',max_length=max_length,pad_token=pad_token,pad_genre_token=pad_genre_token)
val_dataset = GRUDataset(val_history,genre_df=ml_movie_df,mode='eval',max_length=max_length,pad_token=pad_token,pad_genre_token=pad_genre_token)
test_dataset = GRUDataset(test_history,genre_df=ml_movie_df,mode='eval',max_length=max_length,pad_token=pad_token,pad_genre_token=pad_genre_token)

# create the train,validation, and test pytorch dataloader objects
train_dl = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
val_dl = DataLoader(val_dataset,batch_size=64)
test_dl = DataLoader(test_dataset,batch_size=64)


# %%
print("Bert dim: {:d}".format(bert_dim))
print("Genre dim: {:d}".format(genre_dim))
print("Pad Token: {}".format(pad_token))
print("Pad Genre Token: {}".format(pad_genre_token))


# %%
# ------------------Model Initialization----------------------#

# initialize gru4rec model with arguments specified earlier
if model_type == "feature_add":
    model = gru4recF(embedding_dim=embedding_dim,
             hidden_dim=hidden_dim,
             output_dim=output_dim,
             genre_dim=genre_dim,
             batch_first=True,
             max_length=max_length,
             pad_token=pad_token,
             pad_genre_token=pad_genre_token,
             bert_dim=bert_dim,
             tied = tied,
             dropout=dropout)


if model_type == "feature_concat":
    model = gru4recFC(embedding_dim=embedding_dim,
             hidden_dim=hidden_dim,
             output_dim=output_dim,
             genre_dim=genre_dim,
             batch_first=True,
             max_length=max_length,
             pad_token=pad_token,
             pad_genre_token=pad_genre_token,
             bert_dim=bert_dim,
             tied = tied,
             dropout=dropout)

if model_type == "vanilla":
    model = gru4rec_vanilla(hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            batch_first=True,
                            max_length=max_length,
                            pad_token=pad_token,
                            tied=tied,
                            embedding_dim=embedding_dim,
                           device=device)

if model_type =="feature_only":
    model = gru4rec_feature(hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            batch_first=True,
                            max_length=max_length,
                            pad_token=pad_token,
                            bert_dim=bert_dim)

if model_type == "conv":
    model = gru4rec_conv(embedding_dim,
                 hidden_dim,
                 output_dim,
                 batch_first=True,
                 max_length=200,
                 pad_token=0,
                 dropout=0,
                 window=3,
                 tied=tied)
    
if model_type == "nextitnet":
    model = NextItNet(embedding_dim=embedding_dim,
                      output_dim=output_dim,
                      hidden_layers=hidden_layers,
                      dilations=dilations,
                      pad_token=n_items,
                      max_len=max_length)


# %%
if bert_dim != 0:
    model.init_weight(reset_object,feature_embed)
    
model = model.to(device)


# %%
# initialize Adam optimizer with gru4rec model parameters
if train_method != "normal":
    optimizer_features = torch.optim.Adam([param for name,param in model.named_parameters() if (("movie" not in name) or ("plot_embedding" in name) or ("genre" in name)) ],
                                          lr=lr_alternate,weight_decay=reg)
    
    optimizer_ids = torch.optim.Adam([param for name,param in model.named_parameters() if ("plot" not in name) and ("genre" not in name)],
                                     lr=lr,weight_decay=reg)

elif train_method == "normal":
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
    
if freeze_plot and bert_dim !=0:
    model.plot_embedding.weight.requires_grad = False


# %%
if loss_type == "XE":
    loss_fn = nn.CrossEntropyLoss(ignore_index=n_items)
    
elif loss_type == "BPR":
    loss_fn = BPRLoss(user_history = user_history,
                      n_items = n_items, 
                      df = ml_1m,
                      device = device, 
                      samples=num_neg_samples)

elif loss_type == "BPR_MAX":
    loss_fn = BPRMaxLoss(user_history = user_history,
                      n_items = n_items, 
                      df = ml_1m,
                      device = device,
                      reg = reg_bpr,
                      samples=num_neg_samples)


# %%
Recall_Object = Recall_E_prob(ml_1m,user_history,n_users,n_items,k=k,device=device)


# %%
#print("Baseline POP results: ",Recall_Object.popular_baseline())


# %%
#training_hit = Recall_Object(model,train_dl)
#validation_hit = Recall_Object(model,val_dl)
#testing_hit = Recall_Object(model,test_dl)
#print("Training Hits@{:d}: {:.2f}".format(k,training_hit))
#print("Validation Hits@{:d}: {:.2f}".format(k,validation_hit))
#print("Testing Hits@{:d}: {:.2f}".format(k,testing_hit))


# %%
# ------------------Training Initialization----------------------#
max_train_hit = (0,0,0)
max_val_hit = (0,0,0)
max_test_hit = (0,0,0)

max_train_ndcg = (0,0,0)
max_val_ndcg = (0,0,0)
max_test_ndcg = (0,0,0)

max_train_mrr = 0
max_val_mrr = 0
max_test_mrr = 0
i = 0;
for epoch in range(num_epochs):
    print("="*20,"Epoch {}".format(epoch+1),"="*20)
    
    model.train()  
    
    running_loss = 0

    for j,data in enumerate(tqdm(train_dl,position=0,leave=True)):
        
        if train_method != "normal":
            optimizer_features.zero_grad()
            optimizer_ids.zero_grad()
            
        elif train_method == "normal": 
            optimizer.zero_grad()
        
        if genre_dim != 0:            
            inputs,genre_inputs,labels,x_lens,uid = data
            outputs = model(x=inputs.to(device),x_lens=x_lens.squeeze().tolist(),x_genre=genre_inputs.to(device))
        
        elif genre_dim == 0:
            inputs,labels,x_lens,uid = data 
            outputs = model(x=inputs.to(device),x_lens=x_lens.squeeze().tolist())
       
        if tied:
            outputs_ignore_pad = outputs[:,:,:-1]
            if loss_type == "XE":
                loss = loss_fn(outputs_ignore_pad.view(-1,outputs_ignore_pad.size(-1)),labels.view(-1).to(device))
            elif loss_type == "BPR" or loss_type == "BPR_MAX":
                loss = loss_fn(outputs,labels.to(device),x_lens,uid)

            
        else:
            if loss_type == "XE":
                loss = loss_fn(outputs.view(-1,outputs.size(-1)),labels.view(-1).to(device))
            elif loss_type == "BPR" or loss_type == "BPR_MAX":   
                loss = loss_fn(outputs,labels.to(device),x_lens,uid)

        loss.backward()
        
        
        if train_method != "normal":
            if train_method == "interleave":
                # interleave on the epochs
                if (j+1) % 2 == 0:
                    optimizer_features.step()
                else:
                    optimizer_ids.step()

            elif train_method == "alternate":
                if (epoch+1) % 2 == 0:
                    optimizer_features.step()
                else:
                    optimizer_ids.step()
        
    
                    
        elif train_method == "normal":
            optimizer.step()

        running_loss += loss.detach().cpu().item()

    del outputs
    torch.cuda.empty_cache()
    training_hit,training_ndcg,training_mrr = Recall_Object(model,train_dl,"train")
    validation_hit,validation_ndcg,validation_mrr = Recall_Object(model,val_dl,"validation")
    testing_hit,testing_ndcg,testing_mrr = Recall_Object(model,test_dl,"test")
    
    if max_val_mrr < validation_mrr:
        max_val_hit = validation_hit
        max_test_hit = testing_hit
        max_train_hit = training_hit
        
        max_train_ndcg = training_ndcg
        max_val_ndcg = validation_ndcg
        max_test_ndcg = testing_ndcg
        
        max_train_mrr = training_mrr
        max_val_mrr = validation_mrr
        max_test_mrr = testing_mrr
        print("BEST MODEL PERFORMANCE")

    
    torch.cuda.empty_cache()
    print("Training Loss: {:.5f}".format(running_loss/len(train_dl)))
    
    print("Train Hits \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*training_hit))
    print("Train ndcg \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*training_ndcg))
    print("Train mrr \t {:.5f}".format(training_mrr))


    print("Valid Hits \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*validation_hit))
    print("Valid ndcg \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*validation_ndcg))
    print("Valid mrr \t {:.5f}".format(validation_mrr))

    print("Test Hits \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*testing_hit))
    print("Test ndcg \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*testing_ndcg))
    print("Test mrr \t {:.5f}".format(testing_mrr))
    
print("="*100)
print("Maximum Training Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_train_hit))
print("Maximum Validation Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_val_hit))
print("Maximum Testing Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_test_hit))


# %%
print("="*100)
print("Maximum Train Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_train_hit))
print("Maximum Valid Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_val_hit))
print("Maximum Test Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_test_hit))

print("Maximum Train NDCG \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_train_ndcg))
print("Maximum Valid NDCG \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_val_ndcg))
print("Maximum Test NDCG \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_test_ndcg))

print("Maximum Train MRR \t {:.5f}".format(max_train_mrr))
print("Maximum Valid MRR \t {:.5f}".format(max_val_mrr))
print("Maximum Test MRR \t {:.5f}".format(max_test_mrr))

# %% [markdown]
# ##### 

