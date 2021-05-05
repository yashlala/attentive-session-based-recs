# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import argparse
from torch.utils.data import DataLoader
import torch

from preprocessing import *
from dataset import *
from model import *
from utils import bert2dict
"""
TODO:
import metric library 
import objective function library
"""

parser = argparse.ArgumentParser()

# sgd arguments
parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=256)
parser.add_argument('--reg',type=float,help='The regularization strength on l2 norm',default = 0.0)
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=5e-3)
parser.add_argument('--train_method',type=str,help="How you want to switch off between feature optimizers versus item id optimizers ('interleave', 'alternate', 'normal' ...)",default="normal")


# model arguments
parser.add_argument('--embedding_dim',type=int,help="Size of item embedding",default=128)
parser.add_argument('--bert_dim',type=int,help="Size of bert embedding (if 0, then not used, otherwise set to 768",default=0)
parser.add_argument('--hidden_dim',type=int,help="Size of GRU hidden dimension",default=128)
parser.add_argument('--freeze_plot',action='store_true',help='Flag whether to finetune or not, freeze_plot flag means to not finetune')
parser.add_argument('--tied',action='store_true',help='Whether to make the output layer weights the embedding layer weights')
parser.add_argument('--dropout',type=float,help='The dropout rate of output layer of GRU',default=0)

# file name arguments
parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--read_movie_filename',type=str,help='The filename to read all the MovieLens-1 million movie data from to the Dataframe',default="")
parser.add_argument('--read_bert_filename',type=str,help='The filename to read all the pre-computed feature embeddings from or to',default="bert_sequence.txt")


# dataset arguments
parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--max_len',type=int,help='Maximum length for the sequence',default=200)
parser.add_argument('--min_len',type=int,help="Minimum session length for a sequence (filter out sessions less than this",default=10)
parser.add_argument('--size',type=str,help='The dataset (1m , 20m , etc) which you will use',default="20m")




"""
TODO:
ADD ARGUMENTS NEEDED
"""

# ----------------- Variables ----------------------#

# read all arguments from argparse
args = parser.parse_args()

read_filename = args.read_filename
read_bert_filename = args.read_bert_filename
read_movie_filename = args.read_movie_filename


num_epochs = args.num_epochs
lr = args.alpha
batch_size = args.batch_size
reg = args.reg
train_method = args.train_method


hidden_dim = args.hidden_dim
embedding_dim = args.embedding_dim
bert_dim= args.bert_dim
freeze_plot = args.freeze_plot
tied = args.tied
args = args.dropout


k = args.hitsat
max_length = args.max_len
min_len = args.min_len
size = args.size


"""
TODO:
ADD Variable NEEDED
"""
# ------------------Data Initialization----------------------#

# convert .dat file to time-sorted pandas dataframe
ml_1m = create_df(read_filename,size=size)

# remove users who have sessions lengths less than min_len
ml_1m = filter_df(ml_1m,item_min=min_len)

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
    genre_dim = 0
    ml_movie_df = None

# how many unique users, items, ratings and timestamps are there
n_users,n_items,n_ratings,n_timestamp = ml_1m.nunique()

# value that padded tokens shall take
pad_token = n_items

# the output dimension for softmax layer
output_dim = n_items


# get the item id : bert plot embedding dictionary
if bert_dim != 0:
    feature_embed = bert2dict(bert_filename=read_bert_filename)

# create a dictionary of every user's session (history)
# i.e. {user: [user clicks]}
user_history = create_user_history(ml_1m)

# create a dictionary of all items a user has not clicked
# i.e. {user: [items not clicked by user]}
user_noclicks = create_user_noclick(user_history,ml_1m,n_items)

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

# ------------------Model Initialization----------------------#

# initialize gru4rec model with arguments specified earlier
if concat:
    model = gru4recFC(embedding_dim=embedding_dim,
             hidden_dim=hidden_dim,
             output_dim=output_dim,
             genre_dim=genre_dim,
             batch_first=True,
             max_length=max_length,
             pad_token=pad_token,
             pad_genre_token=pad_genre_token,
             bert_dim=bert_dim,
             dropout=dropout,
             tied=tied)
else:
    model = gru4recF(embedding_dim=embedding_dim,
         hidden_dim=hidden_dim,
         output_dim=output_dim,
         genre_dim=genre_dim,
         batch_first=True,
         max_length=max_length,
         pad_token=pad_token,
         pad_genre_token=pad_genre_token,
         bert_dim=bert_dim,
         dropout=dropout,
         tied=tied)

if bert_dim != 0:
    model.init_weight(reset_object,feature_embed)
    
model = model.cuda()

# initialize Adam optimizer with gru4rec model parameters
if train_method != "normal":
    optimizer_features = torch.optim.Adam([param for name,param in model.named_parameters() if (("movie" not in name) or ("plot_embedding" in name) or ("genre" in name)) ],
                                          lr=lr,weight_decay=reg)
    
    optimizer_ids = torch.optim.Adam([param for name,param in model.named_parameters() if ("plot" not in name) and ("genre" not in name)],
                                     lr=lr,weight_decay=reg)

elif train_method == "normal":
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
    
if freeze_plot and bert_dim !=0:
    model.plot_embedding.weight.requires_grad = False
    
# ------------------ Objective/Metric Initialization ------------# 
"""
TODO:
Add the appropriate metric objects/function initializations

Add the appropriate objective objects/function initializations

Create option to choose from arguments which metric/objective combo to use
"""
# ------------------Training Initialization----------------------#

for epoch in range(num_epochs):
    print("="*20,"Epoch {}".format(epoch+1),"="*20)
    
    model.train()  
    
    running_loss = 0

    for j,data in enumerate(train_dl,position=0,leave=True):
        
        if train_method != "normal":
            optimizer_features.zero_grad()
            optimizer_ids.zero_grad()
            
        elif train_method == "normal": 
            optimizer.zero_grad()
        
        if genre_dim != 0:            
            inputs,genre_inputs,labels,x_lens,uid = data
            outputs = model(x=inputs.cuda(),x_lens=x_lens.squeeze().tolist(),x_genre=genre_inputs.cuda())
        
        else:
            inputs,labels,x_lens,uid = data
            outputs = model(x=inputs.cuda(),x_lens=x_lens.squeeze().tolist())

        if tied:
            outputs_ignore_pad = outputs[:,:,:-1]
            loss = """ TODO: ADD LOSS"""
            
        else:
            loss = """ TODO: ADD LOSS"""


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
        
    print("Training Loss: {:.5f}".format(running_loss/len(train_dl)))
    

