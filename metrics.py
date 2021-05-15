# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:13 2021

@author: lpott
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

    
class Recall_E_prob(object):
    def __init__(self,df,user_history,n_users,n_items,k=10,device='cpu'):
        print("="*10,"Creating Hit@{:d} Metric Object".format(k),"="*10)

        self.user_history = user_history
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        
         # get the number of times each item id was clicked across all users
        self.p = df.groupby('item_id',sort='item_id').size()
        self.p = self.p.values / self.p.sum()
        
        self.device=device
            
        
    def __call__(self,model,dataloader,mode="train"):
        
        with torch.no_grad():
            model.eval()

            
            NDCG = 0
            total_hits = 0 
            for data in dataloader:
                
                if model.genre_dim != 0:            
                    inputs,genre_inputs,labels,x_lens,uid = data
                    outputs = model(x=inputs.to(self.device),x_lens=x_lens.squeeze().tolist(),x_genre=genre_inputs.to(self.device))
            
                else:
                    inputs,labels,x_lens,uid = data
                    outputs = model(x=inputs.to(self.device),x_lens=x_lens.squeeze().tolist())
                                
                for i,uid in enumerate(uid.squeeze()):
                    history = self.user_history[uid.item()]
                    
                    if mode == "train":
                        history = set(history[:-2])
                    
                    if mode == "validation":
                        history = set(history[:-1])
                        
                    if mode == "test":
                        history = set(history)
                        
                    sample_negatives = []
                    
                    while len(sample_negatives) < 101:
                        
                        sampled_ids = np.random.choice(self.n_items, 100, replace=False, p=self.p).tolist()
                        sampled_ids = [x for x in sampled_ids if x not in history and x not in sample_negatives]
                        sample_negatives.extend(sampled_ids[:])
                        
                    sample_negatives = sample_negatives[:100].copy()
                    
                    sample_negatives.append(labels[i,x_lens[i].item()-1].item())
                                        
                    topk_items = outputs[i,x_lens[i].item()-1,sample_negatives].argsort(0,descending=True)[:self.k]
                    total_hits += torch.sum(topk_items == 100).item() 
                    rank_of_target = torch.where(topk_items == 100)[0]
                    if rank_of_target.shape[0] > 0:
                        NDCG += 1 / np.log2(rank_of_target.item() + 2)
                    
            del outputs
            del topk_items
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return total_hits/self.n_users*100, NDCG / self.n_users
    
    def popular_baseline(self):
        total_hits = 0 
        NDCG = 0

        for i,uid in enumerate(range(self.n_users)):
            history = self.user_history[uid]
                
            sample_negatives = []
            
            while len(sample_negatives) < 101:
                
                sampled_ids = np.random.choice(self.n_items, 100, replace=False, p=self.p).tolist()
                sampled_ids = [x for x in sampled_ids if x not in history and x not in sample_negatives]
                sample_negatives.extend(sampled_ids[:])
                
            sample_negatives = sample_negatives[:100].copy()
            
            sample_negatives.append(self.user_history[uid][-2])
                                
            topk_items = self.p[np.array(sample_negatives)].argsort()[-self.k:]
            rank_of_target = np.where(topk_items == 100)[0]
            if rank_of_target.shape[0] > 0:
                NDCG += 1 / np.log2(rank_of_target.item() + 2)
            
            total_hits += np.sum(topk_items == 100).item() 
                        
        return total_hits/self.n_users*100, NDCG / self.n_users


class BPRLossWithNoClick(nn.Module):
    """
    BPR loss function that utilizes the no click history object from preprocessing
    
    """
    def __init__(self, user_noclick: dict, device: torch.device, samples=1):
        """
        takes in
        :parameter user_noclick: dictionary output from the no click history function
        :parameter device: device being used (cuda or cpu)
        :parameter samples: amount of negative samples
        """
        
        self.user_noclick = user_noclick
        self.samples = samples
        self.device = device
        super(BPRLossWithNoClick, self).__init__()

    def forward(self, output: torch.Tensor, labels: torch.Tensor, x_lens: torch.Tensor, uids: torch.Tensor):
        """
        takes in the
        :param output: output tensor of model
        :param labels: 2nd element from data tuple of a dataloader (item indices of next item)
        :param x_lens: 3rd element from data tuple (length of seq)
        :param uid: user_ids
        """
        
        accumulator = torch.FloatTensor([0.]).to(self.device)
        for i, uid in enumerate(uids.squeeze()):
            all_indices = output[i, :x_lens[i].item(), :]
            
            positive_scores = torch.gather(all_indices, dim=1, index=labels[i][:x_lens[i].item()].unsqueeze(1))
            
            # sampling with replacement (TODO: might need someone to check me on this)
            # normalize 
            
            negative_item_ids = np.random.choice(self.user_noclick[uid.item()][0][:-2], 
                                                 size=(x_lens[i].item(), self.samples), 
                                                 p=self.user_noclick[uid.item()][1][:-2]/ np.sum(self.user_noclick[uid.item()][1][:-2]))
            
            negative_scores = torch.gather(all_indices, 
                                           dim=1, 
                                           index=torch.LongTensor(negative_item_ids).to(self.device))
            
            
            # negative_scores = output[i, :x_lens[i].item(), negative_item_ids]
            
            difference = positive_scores - negative_scores
            
            # TODO: someone check if i need to actually take a mean of columns then of rows or altogether?
            accumulator += -torch.mean(torch.sum(F.logsigmoid(difference), dim=1))
        
        return accumulator


# class MRR(object):
#     """
#     heavily inspired from this place
#     https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
#     """
#     def __init__(self, device: torch.device, k=10):
#         """
#         seems like the only thing we need to keep track of is the device
#         and the @hits
#         """
#         print("="*10,"Creating MRR@{:d} Metric Object".format(k),"="*10)
#         self.device = device
#         self.number = k
        
        
#     def __call__(self, model, dataloader):
#         """
#         only requires the labels since that seems the only thing it needs for calculation

#         BEWARE: implementation needs improvement or shouldn't be called on large dataloaders
#         """
        
#         with torch.no_grad():
#             model.eval()
            
#             MRR_count = 0
#             iters = 0
            
#             for data in dataloader:
                
#                 if model.genre_dim != 0:            
#                     inputs,genre_inputs,labels,x_lens,uid = data
#                     outputs = model(x=inputs.to(self.device),
#                                     x_lens=x_lens.squeeze().tolist(),
#                                     x_genre=genre_inputs.to(self.device))
            
#                 else:
#                     inputs,labels,x_lens,uid = data
#                     outputs = model(x=inputs.to(device),x_lens=x_lens.squeeze().tolist())
                    
#                 # this is the part that takes forever (takes a couple of secs on cpu)
#                 indices_sorted = torch.argsort(outputs, dim=-1)
#                 indices_to_check = indices_sorted[:, :, :self.number]
                
#                 #formatting the labels so it's an easy boolean comparison
#                 tmp_label_view = labels.unsqueeze(2)
#                 label_indices = tmp_label_view.expand_as(indices_to_check)
                
#                 hits = (indices_to_check == label_indices).nonzero()
#                 ranks = hits[:, -1] + 1
#                 recip_rank = torch.reciprocal(ranks)
#                 mrr = torch.sum(recip_rank).item() / labels.size(0)
                
                
#                 MRR_count += mrr
#                 iters += 1
                
#             del outputs
#             del indices_sorted
#             del hits
#             del ranks
                
#             return MRR_count / iters