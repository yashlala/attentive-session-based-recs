# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:13 2021

@author: lpott, hamlin
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

    
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
        """
        
        Return
        3 - tuple of hits
        3 - tuple of ndcg
        1 integer that is the total MRR
        """
        
        with torch.no_grad():
            model.eval()

            
            ndcg_1 = 0
            hit_1 = 0
            
            ndcg_5 = 0
            hit_5 = 0
            
            ndcg_10 = 0
            hit_10 = 0

            mrr = 0
            
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
                        
                    
                    #sample_negatives = []
                    sample_negatives = [labels[i,x_lens[i].item()-1].item()]
                    while len(sample_negatives) < 101:
                        
                        sampled_ids = np.random.choice(self.n_items, 100, replace=False, p=self.p).tolist()
                        sampled_ids = [x for x in sampled_ids if x not in history and x not in sample_negatives]
                        sample_negatives.extend(sampled_ids[:])
                        
                    #sample_negatives = sample_negatives[:100].copy()
                    sample_negatives = sample_negatives[:101].copy()
                    #sample_negatives.append(labels[i,x_lens[i].item()-1].item())
                    

                    


                        

                    rank = torch.where(outputs[i,x_lens[i].item()-1,sample_negatives].argsort(descending=True)==0)[0].item()          
                    #topk_items = outputs[i,x_lens[i].item()-1,sample_negatives].argsort(0,descending=True)[:self.k]
                    
                    
                    
                    if rank < 1:
                        ndcg_1 += 1
                        hit_1 += 1
                    if rank < 5:
                        ndcg_5 += 1 / np.log2(rank + 2)
                        hit_5 += 1
                    if rank < 10:
                        ndcg_10 += 1 / np.log2(rank + 2)
                        hit_10 += 1
                    
                    mrr += 1 / (rank + 1)
                    #total_hits += torch.sum(topk_items == 100).item() 
                    #rank_of_target = torch.where(topk_items == 100)[0]
                    #if rank_of_target.shape[0] > 0:
                    #    NDCG += 1 / np.log2(rank_of_target.item() + 2)
                   
            del outputs
            del rank
            del sample_negatives
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return (hit_10/self.n_users,hit_5/self.n_users,hit_1/self.n_users),(ndcg_10/self.n_users,ndcg_5/self.n_users,ndcg_1/self.n_users), mrr / self.n_users
    
    def popular_baseline(self):
        """
        NOTE: MRR is off by 0.01 from the baseline in the BERT4REC paper
        """
        total_hits = 0 
        NDCG = 0

        MRR = 0

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
            all_items = self.p[np.array(sample_negatives)].argsort()
            rank_of_target = np.where(topk_items == 100)[0]

            all_rank_of_target = np.where(all_items == 100)[0]
            if rank_of_target.shape[0] > 0:
                NDCG += 1 / np.log2(rank_of_target.item() + 2)
                
            
            total_hits += np.sum(topk_items == 100).item() 

            MRR += 1 / (all_rank_of_target + 1)


            
                        
        return total_hits/self.n_users*100, NDCG / self.n_users, MRR / self.n_users


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
        
        return accumulator / uids.shape[0]


class BPRLoss(nn.Module):
    """
    BPR loss function that does not utilize the no click history from preprocessing
    
    """
    
    def __init__(self, user_history: dict, n_items, df: pd.DataFrame, device: torch.device, samples=1):
        self.n_items = n_items
        self.samples = samples
        self.user_history = user_history
        
        # same from recall
        self.p = df.groupby('item_id',sort='item_id').size()
        self.p = self.p.values / self.p.sum()
        
        print(self.p.size)
        
        self.device = device
        super(BPRLoss, self).__init__()
        
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
            uid_history = set(self.user_history[uid.item()][:-2]) # we only compute loss when training
            all_indices = output[i, :x_lens[i].item(), :]
            
            positive_scores = torch.gather(all_indices, dim=1, index=labels[i][:x_lens[i].item()].unsqueeze(1))
            
            negative_item_ids = []
            
            # check if we match the sequence length
            while len(negative_item_ids) < x_lens[i].item():
                inner_neg_ids = []
                
                # TODO: check this if sampling is correct
                # check if we match the sample length
                while len(inner_neg_ids) < self.samples:
                    sampled_ids = np.random.choice(self.n_items, self.samples, replace=False, p=self.p).tolist()
                    sampled_ids = [x for x in sampled_ids if x not in uid_history and x not in inner_neg_ids]
                    sampled_ids = sampled_ids[:self.samples - len(inner_neg_ids)]
                    inner_neg_ids.extend(sampled_ids[:])
                
                #add in one sequence
                negative_item_ids.append(np.array(inner_neg_ids))
                
            negative_scores = torch.gather(all_indices, 
                                           dim=1, 
                                           index=torch.LongTensor(np.array(negative_item_ids)).to(self.device))
            
            difference = positive_scores - negative_scores
            
            
            accumulator += -torch.mean(torch.sum(F.logsigmoid(difference), dim=1))
        
        return accumulator / uids.shape[0]



def gen_negative_samples(b, users, user_history, N_s, N, p): 
    """TODO

    Params: 
        b := the batch size
        users := (b, 1) array of users whose IDs TODO

    Returns: 
        (b, N_s) array. Each row is 
    """
    probs = np.broadcast_to(p, (b, N))
    for i, u in enumerate(users): 
        for click in user_history[u]: 
            probabilities[i, click] = 0
    
    ret = np.zeros((b, N_s))
    for i in len(probs): 
        ret[i] = np.random.choice(N, size=(N_s,), p=probs[i], replace=False)
    return ret


class BPRMaxLoss(nn.Module):
    """
    BPR loss function that does not utilize the no click history from preprocessing
    
    """
    
    def __init__(self, user_history: dict, n_items, df: pd.DataFrame,
            device: torch.device, reg = 0.0, samples=1):
        self.n_items = n_items
        self.samples = samples
        self.user_history = user_history
        
        # same from recall
        self.p = df.groupby('item_id',sort='item_id').size()
        self.p = self.p.values / self.p.sum()
        
        print(self.p.size)
        
        self.device = device
        self.reg = reg
        super(BPRMaxLoss, self).__init__()

        
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
            uid_history = set(self.user_history[uid.item()][:-2]) # we only compute loss when training
            all_indices = output[i, :x_lens[i].item(), :]
            
            positive_scores = torch.gather(all_indices, dim=1, index=labels[i][:x_lens[i].item()].unsqueeze(1))
            
            
            negative_item_ids = []
            
            # check if we match the sequence length
            while len(negative_item_ids) < x_lens[i].item():
                
                inner_neg_ids = []
                    
                # check if we match the sample length
                while len(inner_neg_ids) < self.samples:
                    sampled_ids = np.random.choice(self.n_items, self.samples, replace=False, p=self.p).tolist()
                    sampled_ids = [x for x in sampled_ids if x not in uid_history and x not in inner_neg_ids]
                    sampled_ids = sampled_ids[:self.samples - len(inner_neg_ids)]
                    inner_neg_ids.extend(sampled_ids[:])
                
                #add in one sequence
                negative_item_ids.append(np.array(inner_neg_ids))
                
                """
                while len(inner_neg_ids) < self.samples+1:
                    sampled_ids = np.random.choice(self.n_items, self.samples, replace=False, p=self.p).tolist()
                    sampled_ids = [x for x in sampled_ids if x not in uid_history and x not in inner_neg_ids]
                    inner_neg_ids.extend(sampled_ids[:])
                        
                inner_neg_ids = inner_neg_ids[:self.samples].copy()
                negative_item_ids.append(np.array(inner_neg_ids))
                """


            negative_scores = torch.gather(all_indices, 
                                           dim=1, 
                                           index=torch.LongTensor(np.array(negative_item_ids)).to(self.device))
            
            difference = positive_scores - negative_scores
            
            #TODO: check this stuff
            soft_max_neg = F.softmax(negative_scores, dim=1) 
            
            #TODO: check this stuff
            accumulator += torch.mean(
                -torch.log(torch.sum(soft_max_neg * torch.sigmoid(difference), dim=1)) 
                + self.reg * torch.sum(soft_max_neg * negative_scores**2, dim = 1))

        return accumulator / uids.shape[0]