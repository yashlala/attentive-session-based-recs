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
from metrics import *
"""
TODO:
import metric library 
import objective function library
"""

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=0.001)
parser.add_argument('--embedding_dim',type=int,help="Size of item embedding",default=64)
parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--read_movie_filename',type=str,help='The filename to read all the MovieLens-1 million movie data from to the Dataframe',default="")
parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=32)
parser.add_argument('--reg',type=float,help='The regularization strength on l2 norm',default = 0.0)
parser.add_argument('--hidden_layers',type=int,help="The number of hidden layeres",default=2)
parser.add_argument('--dilations',type=str,help="The dilation scheme of the hidden layers",default="1,2,4,8")
#parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--max_len',type=int,help='Maximum length for the sequence',default=200)
parser.add_argument('--min_len',type=int,help="Minimum session length for a sequence (filter out sessions less than this",default=10)
parser.add_argument('--size',type=str,help='The dataset (1m , 20m , etc) which you will use',default="20m")

# ----------------- Variables ----------------------#

# read all arguments from argparse
args = parser.parse_args()

read_filename = args.read_filename
read_movie_filename = args.read_movie_filename
size = args.size

num_epochs = args.num_epochs
lr = args.alpha
batch_size = args.batch_size
reg = args.reg

embedding_dim = args.embedding_dim
hidden_layers = args.hidden_layers
dilations= list(map(int,args.dilations.split(",")))

k = args.hitsat
max_length = args.max_len
min_len = args.min_len
# ------------------Data Initialization----------------------#

# convert .dat file to time-sorted pandas dataframe
ml_1m = create_df(read_filename,size=size)

# remove users who have sessions lengths less than min_len
ml_1m = filter_df(ml_1m,item_min=min_len)

# initialize reset object
reset_object = reset_df()

# map all user ids and item ids to range 0 - Number of Users/Items 
# i.e. [1,7,5] -> [0,2,1]
ml_1m = reset_object.fit_transform(ml_1m)

# how many unique users, items, ratings and timestamps are there
n_users,n_items,n_ratings,n_timestamp = ml_1m.nunique()

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

# value that padded tokens shall take
pad_token = n_items


# initialize the train,validation, and test pytorch dataset objects
# eval pads all items except last token to predict
train_dataset = GRUDataset(train_history,mode='train',max_length=max_length,pad_token=pad_token)
val_dataset = GRUDataset(val_history,mode='eval',max_length=max_length,pad_token=pad_token)
test_dataset = GRUDataset(test_history,mode='eval',max_length=max_length,pad_token=pad_token)

# the output dimension for softmax layer
output_dim = n_items

# create the train,validation, and test pytorch dataloader objects
train_dl = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
val_dl = DataLoader(val_dataset,batch_size=64)
test_dl = DataLoader(test_dataset,batch_size=64)
# ------------------Model Initialization----------------------#

# initialize NextItNet model with arguments specified earlier
model = NextItNet(embedding_dim,output_dim,hidden_layers=hidden_layers,dilations=dilations,pad_token=n_items,max_len=max_length).cuda()

# initialize Adam optimizer with gru4rec model parameters
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
# ------------------Objective/Metric Initialization -------------#
"""
TODO:
Add the appropriate metric objects/function initializations

Add the appropriate objective objects/function initializations

Create option to choose from arguments which metric/objective combo to use
"""
# ------------------Training Initialization----------------------#

max_train_hit = 0
max_val_hit = 0
max_test_hit = 0

for epoch in range(num_epochs):
    print("="*20,"Epoch {}".format(epoch+1),"="*20)
    
    model.train()  
    
    running_loss = 0

    for data in train_dl:
        optimizer.zero_grad()

        inputs,labels,x_lens,uid = data
        
        outputs = model(inputs.cuda())
        """
        TODO:
        Add loss = objective function initialized
        loss = previously loss_fn(outputs.view(-1,outputs.size(-1)),labels.view(-1).cuda())
        """

        loss.backward()
        optimizer.step()
        
        running_loss += loss.detach().cpu().item()

    
    print("Training Loss: {:.5f}".format(running_loss/len(train_dl)))