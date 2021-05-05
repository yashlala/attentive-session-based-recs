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
from metrics import *

from time import time
"""
TODO:
import metric library 
import objective function library
"""

parser = argparse.ArgumentParser()

# file name arguments
parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--read_movie_filename',type=str,help='The filename to read all the MovieLens-1 million movie data from to the Dataframe',default="")
parser.add_argument('--read_bert_filename',type=str,help='The filename to read all the pre-computed feature embeddings from or to',default="bert_sequence.txt")

# model thing ... 
parser.add_argument('--freeze_plot',action='store_true',help='Flag whether to finetune or not, freeze_plot flag means to not finetune')
parser.add_argument('--tied',action='store_true',help='Whether to make the output layer weights the embedding layer weights')



# dataset arguments
parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--max_len',type=int,help='Maximum length for the sequence',default=200)
parser.add_argument('--min_len',type=int,help="Minimum session length for a sequence (filter out sessions less than this",default=10)
parser.add_argument('--size',type=str,help='The dataset (1m , 20m , etc) which you will use',default="1m")

parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=256)

"""
TODO:
ADD ARGUMENTS NEEDED
"""

# ----------------- Variables ----------------------#
torch.cuda.empty_cache()

# read all arguments from argparse
args = parser.parse_args()

read_filename = args.read_filename
read_bert_filename = args.read_bert_filename
read_movie_filename_ = args.read_movie_filename


k = args.hitsat
max_length = args.max_len
min_len = args.min_len
size = args.size

batch_size = args.batch_size
freeze_plot = args.freeze_plot
tied = args.tied

train_method_grid = ["alternate","normal"]
reg_grid = [0,1e-4,1e-5]
lr_grid = [5e-3,1e-3,1e-4]
num_epochs_grid = [50]
hidden_dim_grid = [256]
embedding_dim_grid = [256]
bert_dim_grid = [768,0]
read_movie_filename_grid = ["",read_movie_filename_]

"""
TODO:
ADD Variable NEEDED
"""
search_file = open("grid_search{:d}.txt".format(k),"w")
time_start = time()
search_file.write("Epoch,Bert_dim,Genre_dim,Embed_dim,Hidden_dim,reg,lr,train_method,movie_file_name,concat,batch_size\n")
print("Epoch,Bert_dim,Genre_dim,Embed_dim,Hidden_dim,reg,lr,train_method,movie_file_name,concat,batch_size\n")

# ------------------Data Initialization----------------------#
for read_movie_filename in read_movie_filename_grid:
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
   
        
    # create a dictionary of every user's session (history)
    # i.e. {user: [user clicks]}
    user_history = create_user_history(ml_1m)
    
    # Create the recall object... useful to get the negative sampling etc...
    Recall_Object = Recall_E_prob(ml_1m,user_history,n_users,n_items,k=k)
    
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
    
    for bert_dim in bert_dim_grid:
        # get the item id : bert plot embedding dictionary
        if bert_dim != 0:
            feature_embed = bert2dict(bert_filename=read_bert_filename)    
        # ------------------Model Initialization----------------------#
        
        # initialize gru4rec model with arguments specified earlier
        for concat in [True, False]:
            for num_epochs in num_epochs_grid:
                for reg in reg_grid:
                    for lr in lr_grid:
                        for embedding_dim in embedding_dim_grid:
                            for hidden_dim in hidden_dim_grid:
                                for train_method in train_method_grid:

                                    if concat:
                                        model = gru4recFC(embedding_dim=embedding_dim,
                                                 hidden_dim=hidden_dim,
                                                 output_dim=output_dim,
                                                 genre_dim=genre_dim,
                                                 batch_first=True,
                                                 max_length=max_length,
                                                 pad_token=pad_token,
                                                 pad_genre_token=pad_genre_token,
                                                 bert_dim=bert_dim)
                                    else:
                                        model = gru4recF(embedding_dim=embedding_dim,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             genre_dim=genre_dim,
                                             batch_first=True,
                                             max_length=max_length,
                                             pad_token=pad_token,
                                             pad_genre_token=pad_genre_token,
                                             bert_dim=bert_dim)
                                    
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
                                    loss_fn = nn.CrossEntropyLoss(ignore_index=n_items)
    
                                    # ------------------Training Initialization----------------------#
                                    print("="*100)
                                    print("Epoch,Bert_dim,Genre_dim,Embed_dim,Hidden_dim,reg,lr,train_method,movie_file_name,concat,batch_size")
                                    print("{},{},{},{},{},{},{},{},{},{},{}".format(num_epochs,bert_dim,genre_dim,embedding_dim,hidden_dim,reg,lr,train_method,read_movie_filename,concat,batch_size))
                                    
                                    max_train_hit = 0
                                    max_val_hit = 0
                                    max_test_hit = 0
                                    
                                    time_optimize=time()
                                    for epoch in range(num_epochs):                                        
                                        model.train()  
                                                                        
                                        for j,data in enumerate(train_dl):
                                            
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
                                                loss = loss_fn(outputs_ignore_pad.view(-1,outputs_ignore_pad.size(-1)),labels.view(-1).cuda())
                                                
                                            else:
                                                loss = loss_fn(outputs.view(-1,outputs.size(-1)),labels.view(-1).cuda())
                                    
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
                                                
                                        training_hit = Recall_Object(model,train_dl)
                                        validation_hit = Recall_Object(model,val_dl)
                                        testing_hit = Recall_Object(model,test_dl)
                                        
                                        if max_val_hit < validation_hit:
                                            max_val_hit = validation_hit
                                            max_test_hit = testing_hit
                                            max_train_hit = training_hit
                                            
                                    time_optimize = time()-time_optimize
                                    print("Maximum Training Hit@{:d}: {:.2f}".format(k,max_train_hit))
                                    print("Maximum Validation Hit@{:d}: {:.2f}".format(k,max_val_hit))
                                    print("Maximum Testing Hit@{:d}: {:.2f}".format(k,max_test_hit))
                                    print("Search Time: {:.2f}".format(time_optimize))  
                                    
                                    search_file.write("="*100+"\n")
                                    search_file.write("{},{},{},{},{},{},{},{},{},{},{}".format(num_epochs,bert_dim,genre_dim,embedding_dim,hidden_dim,reg,lr,train_method,read_movie_filename,concat,batch_size))
                                    search_file.write("\n")
                                    search_file.write("{:.2f},{:.2f},{:.2f}".format(max_train_hit,max_val_hit,max_test_hit))
                                    search_file.write("\n")
            
                                    torch.cuda.empty_cache()
time_end = time()
total_time = time_end - time_start

print("Hyperparameter Search Time: {:.2f}".format(total_time))
search_file.close()