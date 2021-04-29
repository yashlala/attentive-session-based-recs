# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:41:47 2021

@author: lpott
"""

import torch
from torch.utils.data import Dataset

class GRUDataset(Dataset):
    def __init__(self, u2seq, mode='train',max_length=200,pad_token=0):
        """
        Parameters
        ----------
        u2seq : dictionary
            Dictionary containing all user's session history. Key is user id, value is list of user clicks [102, 1203 ...] .
        mode : string, optional
            Whether the dataloader is used for training "train" or for validation/testing "eval". The default is 'train'.
            If eval, uses only last item in sequence for label
        max_length : integer, optional
            The maximum length a user session may be. The default is 200.
        pad_token : integer, optional
            The padding value to use when padding is applied to sessions to keep a given session length equal to the maximum length. The default is 0.

        Returns
        -------
        None.

        """
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_length
        self.pad_token = pad_token
        self.mode=mode

    def __len__(self):
        """
        Returns
        -------
            The number of users in dataset for training/evaluating/testing.
        """
        return len(self.users)

    def __getitem__(self, index):
        """
        

        Parameters
        ----------
        index : integer
            The user id to get the user's session history..

        Returns
        -------
        Tuple of form (user session inputs, user session labels, user session length, user id)
        
        user session inputs: torch LongTensor
            The sequence of clicks in a user session (only length of maximum length).
        user session labels: torch LongTensor
            The sequence of clicks in a user session (only length of maximum length) offset by 1 to use as labels.
        user session length: torch LongTensor
            The length before padding of the user session history (if user session history is greater than maximum length then length=maximum length).
        user id: torch LongTensor
            The user id
        """
        
        # get the user id
        user = self.users[index]
        
        # get the user session history 
        seq = self._getseq(user)
        
        
        if self.mode == 'train':
            # input for training is all items in session history except last one
            tokens = seq[:-1]
            # label for training is all items in session history except first (offset of tokens by 1)
            labels = seq[1:]

            # get the session length of the tokens, and the session length of the labels
            # if session length > maximum length it is equal to maximum length
            x_len = len(tokens)
            y_len = len(labels)

            # how many pad tokens are needed to make the input sequence and output sequence
            # length equal to the maximum length 
            x_mask_len = self.max_len - x_len
            y_mask_len = self.max_len - y_len

            # append the pad tokens to the end of the input sequence and the output sequence
            tokens =  tokens + [self.pad_token] * x_mask_len 
            labels =  labels + [self.pad_token] * y_mask_len

        
        if self.mode == 'eval':
            tokens = seq[:-1]
            
            # only difference between 'train' : the label is only the last item in the session history
            labels = seq[-1:]
            
            x_len = len(tokens)
            
            labels = [self.pad_token] * (x_len-1) + labels
            
            y_len = len(labels)


            x_mask_len = self.max_len - x_len
            y_mask_len = self.max_len - y_len

            tokens =  tokens + [self.pad_token] * x_mask_len 
            labels =  labels + [self.pad_token] * y_mask_len
        
        return torch.LongTensor(tokens), torch.LongTensor(labels),torch.LongTensor([x_len]),torch.LongTensor([user])

    def _getseq(self, user):
        """
        Parameters
        ----------
        user : integer
            The user id.

        Returns
        -------
        list
            The corresponding user's session history.
        """
        return self.u2seq[user]