# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
#!pip install transformers

# %% [markdown]
# # Tested Hyperparameters

# %%
# All Hyperparameters in the Spreadsheet.

num_epochs_all = [50]
lr_all = [ 0.01, 0.001, 0.005, 0.0005]
# used when using two optimizers ("alternate" training method). 
# Usually set to lr / 10, but try tweaking it. 
lr_alt_all = [ 0.01, 0.001, 0.005, 0.0005]
batch_size =  64
reg_all = [0,5e-3,5e-4]
bpr_reg_all = [0.8,2,0]
num_neg_samples_all = [5,25,100]
train_method_all = [ "alternate","interleave"]
hidden_dim_all = [256,128 ]
embedding_dim_all = [256,128 ] 
bert_dim_all = [ 768 ]
# : ???
max_length = 200 
freeze_plot_all = [True, False ] 
tied_all = [ False ] 
loss_type_all = [ "BPR_MAX" ] 
dilations_all = [ (1,2,2,4) ] # Only used for cross entropy.

# Hyperparameters not in the Spreadsheet: 

window = 3
dropout= 0
k = 10
min_len = 10

# NextItNet options. 
hidden_layers = 3
model_type = "feature_add"

# %% [markdown]
# # Data Loading and Preprocessing

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing import *
from dataset import *
from metrics import *
from model import *
from utils import bert2dict


# %%
read_filename ="data/movielens-1m/ratings.dat"
read_bert_filename = "data/bert_sequence_1m.txt"
read_movie_filename = "" 
size = "1m"
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


# %%
# ------------------Data Initialization----------------------#
# convert .dat file to time-sorted pandas dataframe
ml_1m = create_df(read_filename, size=size)

# remove users who have session lengths less than min_len
ml_1m = filter_df(ml_1m, item_min=min_len)

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
n_users, n_items, n_ratings, n_timestamp = ml_1m.nunique()

# value that padded tokens shall take
pad_token = n_items

# the output dimension for softmax layer
output_dim = n_items

# get the item id : bert plot embedding dictionary
feature_embed = bert2dict(bert_filename=read_bert_filename)
		
# create a dictionary of every user's session (history)
# i.e. {user: [user clicks]}
user_history = create_user_history(ml_1m)


# %%
# split data by leave-one-out strategy
# have train dictionary {user: [last 41 items prior to last 2 items in user session]}
# have val dictionary {user: [last 41 items prior to last item in user session]}
# have test dictionary {user: [last 41 items]}
# i.e. if max_length = 4, [1,2,3,4,5,6] -> [1,2,3,4] , [2,3,4,5] , [3,4,5,6]
train_history,val_history,test_history = train_val_test_split(user_history,max_length=max_length)

# initialize the train,validation, and test pytorch dataset objects
# eval pads all items except last token to predict
train_dataset = GRUDataset(train_history,genre_df=ml_movie_df,mode='train',max_length=max_length,
													 pad_token=pad_token,pad_genre_token=pad_genre_token)
val_dataset = GRUDataset(val_history,genre_df=ml_movie_df,mode='eval',max_length=max_length,
												 pad_token=pad_token,pad_genre_token=pad_genre_token)
test_dataset = GRUDataset(test_history,genre_df=ml_movie_df,mode='eval',max_length=max_length,
													pad_token=pad_token,pad_genre_token=pad_genre_token)

# create the train,validation, and test pytorch dataloader objects
train_dl = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
val_dl = DataLoader(val_dataset,batch_size=128)
test_dl = DataLoader(test_dataset,batch_size=128)

# %% [markdown]
# # Model Initialization and Training Functions

# %%
def initialize_model(model_type, device,
						 embedding_dim,
						 hidden_dim,
						 output_dim,
						 genre_dim,
						 bert_dim,
						 max_length,
						 tied,
						 batch_first=True,
						 pad_token=pad_token,
						 pad_genre_token=pad_genre_token,
						 dropout=dropout): 
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
								 window=window,
								 tied=tied)
		
		if model_type == "nextitnet":
				model = NextItNet(embedding_dim=embedding_dim,
											output_dim=output_dim,
											hidden_layers=hidden_layers,
											dilations=dilations,
											pad_token=n_items,
											max_len=max_length)
		
		if bert_dim != 0:
				model.init_weight(reset_object,feature_embed)
		
		model = model.to(device)
		return model


# %%
# TODO: move tihs somewhere
# if freeze_plot and bert_dim != 0:
#    model.plot_embedding.weight.requires_grad = False


# %%
def initialize_loss_function(loss_type, n_neg_samples, bpr_reg):
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
											reg = bpr_reg, 
											samples=num_neg_samples)
		else: 
				raise ValueError("Unknown Loss Type.")
				
		return loss_fn


# %%
# Initialize Metric Object
Recall_Object = Recall_E_prob(ml_1m,user_history,n_users,n_items,k=k,device=device)


# %%
# ------------------Training Initialization----------------------#


def record_best_tuple(max_train, max_validation, max_testing, new_train, new_validation, new_testing): 
		if max_validation[0] < new_validation[0]: 
				return new_train, new_validation, new_testing
		return max_train, max_validation, max_testing


def train_model(model, num_epochs, loss_fn, loss_type, train_method, tied, lr, lr_alternate, reg): 
		max_train_hit = (0,0,0)
		max_val_hit = (0,0,0)
		max_test_hit = (0,0,0)

		max_train_ndcg = (0,0,0)
		max_val_ndcg = (0,0,0)
		max_test_ndcg = (0,0,0)

		max_train_mrr = 0
		max_val_mrr = 0
		max_test_mrr = 0

		training_hit = (0,0,0)
		training_ndcg = (0,0,0)
		training_mrr = (0,0,0)

		testing_hit = (0,0,0)
		testing_ndcg = (0,0,0)
		testing_mrr = (0,0,0)
		
		if train_method != "normal":
				optimizer_features = torch.optim.Adam([param for name, param in model.named_parameters() 
																							 if (("movie" not in name) or ("plot_embedding" in name) 
																							 or ("genre" in name))],
																							lr=lr_alternate,weight_decay=reg)
				optimizer_ids = torch.optim.Adam([param for name, param in model.named_parameters() 
																					if ("plot" not in name) and ("genre" not in name)],
																				 lr=lr,weight_decay=reg)
		else:
				optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
		

		for epoch in tqdm(range(num_epochs),position=0,leave=True):
		
				model.train()  
		
				running_loss = 0

				for j, data in enumerate(train_dl):
						if train_method != "normal":
								optimizer_features.zero_grad()
								optimizer_ids.zero_grad()
						else: 
								optimizer.zero_grad()
				
						if genre_dim != 0:            
								inputs, genre_inputs, labels, x_lens,uid = data
								outputs = model(x=inputs.to(device),x_lens=x_lens.squeeze().tolist(),
																x_genre=genre_inputs.to(device))
						else:
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
										
						else:
								optimizer.step()

						running_loss += loss.detach().cpu().item()

				del outputs
		
				if torch.cuda.is_available(): 
						torch.cuda.empty_cache()
				
				validation_hit, validation_ndcg, validation_mrr = Recall_Object(model,val_dl,"validation")

				if validation_mrr > max_val_mrr: 
						# Get the training and ndcg values
						training_hit, training_ndcg, training_mrr = Recall_Object(model,train_dl,"train")
						testing_hit, testing_ndcg, testing_mrr = Recall_Object(model,test_dl,"test")


						max_train_mrr = training_mrr
						max_val_mrr = validation_mrr
						max_test_mrr = testing_mrr

						# Record the best metrics that our model obtained
						max_train_ndcg, max_val_ndcg, max_test_ndcg = record_best_tuple(
								max_train_ndcg, max_val_ndcg, max_test_ndcg, 
								training_ndcg, validation_ndcg, testing_ndcg)
						max_train_hit, max_val_hit, max_test_hit = record_best_tuple(
								max_train_hit, max_val_hit, max_test_hit, 
								training_hit, validation_hit, testing_hit)
		
				if torch.cuda.is_available():
						torch.cuda.empty_cache()
							
		print("="*100)
		print("Maximum Training Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_train_hit))
		print("Maximum Validation Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_val_hit))
		print("Maximum Testing Hit \t @10: {:.5f} \t @5 : {:.5f} \t @1 : {:.5f}".format(*max_test_hit))
		return ((max_train_hit, max_val_hit, max_test_hit), 
						(max_train_ndcg, max_test_ndcg, max_val_ndcg), 
						(max_train_mrr, max_test_mrr, max_val_mrr))

# %% [markdown]
# # Training Loop

# %%
import json
	
for num_epochs in num_epochs_all:
	for lr in lr_all:
		for lr_alt in lr_alt_all:
			for reg, bpr_reg in zip(reg_all, bpr_reg_all):
				for num_neg_samples in num_neg_samples_all:
					for train_method in train_method_all: 
						for hidden_dim in hidden_dim_all: 
							for embedding_dim in embedding_dim_all:
								for bert_dim in bert_dim_all: 
									for freeze_plot in freeze_plot_all: 
										for loss_type in loss_type_all: 
											for dilations in dilations_all: 
												for tied in tied_all: 
													model = initialize_model(
															model_type=model_type,
															device=device, 
															embedding_dim=embedding_dim,
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

													loss_function = initialize_loss_function(loss_type, num_neg_samples, bpr_reg)
													assert loss_function is not None

													((max_train_hit, max_val_hit, max_test_hit), 
														(max_train_ndcg, max_test_ndcg, max_val_ndcg), 
														(max_train_mrr, max_test_mrr, max_val_mrr)) = (
															train_model(model, num_epochs, loss_function, loss_type, train_method, tied, 
																					lr, lr_alt, reg))


													row_params = {
														'num_epochs': num_epochs,
														'lr': lr, 
														'lr_alt': lr_alt, 
														'reg': reg,
														'bpr_reg': bpr_reg,
														'train_method': train_method,
														'hidden_dim': hidden_dim,
														'embedding_dim': embedding_dim,
														'bert_dim': bert_dim,
														'max_length': max_length,
														'freeze_plot': freeze_plot,
														'loss_type': loss_type,
														'dilations': dilations
													}

													
													row_results = { 
														'max_train_hit': max_train_hit,
														'max_val_hit': max_val_hit,
														'max_test_hit': max_test_hit, 
														'max_train_ndcg': max_train_ndcg, 
														'max_test_ndcg': max_test_ndcg, 
														'max_val_ndcg': max_val_ndcg, 
														'max_train_mrr': max_train_mrr, 
														'max_test_mrr': max_test_mrr, 
														'max_val_mrr': max_val_mrr, 
													}
													row_entry = { 
															'params': row_params, 
															'results': row_results
													}
													with open("hyperparam-results.txt", 'a') as f: 
														json.dump(row_entry, f)


