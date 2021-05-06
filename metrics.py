# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:13 2021

@author: lpott
"""
import torch
import numpy as np
from tqdm import tqdm

    
class Recall_E_prob(object):
    def __init__(self,df,user_history,n_users,n_items,k=10):
        print("="*10,"Creating Hit@{:d} Metric Object".format(k),"="*10)

        self.user_history = user_history
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        
         # get the number of times each item id was clicked across all users
        self.p = df.groupby('item_id',sort='item_id').size()
        self.p = self.p.values / self.p.sum()
            
        
    def __call__(self,model,dataloader,mode="train"):
        
        model.eval()
        with torch.no_grad():
            
            total_hits = 0 
            for data in dataloader:
                
                if model.genre_dim != 0:            
                    inputs,genre_inputs,labels,x_lens,uid = data
                    outputs = model(x=inputs.cuda(),x_lens=x_lens.squeeze().tolist(),x_genre=genre_inputs.cuda())
            
                else:
                    inputs,labels,x_lens,uid = data
                    outputs = model(x=inputs.cuda(),x_lens=x_lens.squeeze().tolist())
                                
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
                    total_hits += torch.sum(topk_items == 100).cpu().item() 
                    

                #torch.cuda.empty_cache()
                
        return total_hits/self.n_users*100
    
    def popular_baseline(self):
        total_hits = 0 

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
            total_hits += np.sum(topk_items == 100).item() 
                        
        return total_hits/self.n_users*100