# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:39:46 2021

@author: lpott, hamlinliu17
"""
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder 


# TODO: check if you can increase the efficiency

def create_df(filename=None,size="1m"):
    """
    Parameters
    ----------
    filename : string, optional
        The filename (and path) to the .dat file containing users, items, ratings, and timestamps. The default is None.

    Returns
    -------
    pandas dataframe
        returns a sorted by timestamp pandas dataframe with 4 columns: user id, item id, rating, and timestamp.
    """
    print("="*10,"Creating DataFrame","="*10)
    if size == "1m":
        # read in the .dat file, and the entries on each line are separated by '::'
        df = pd.read_csv(filename,sep='::',header=None)
        df.columns= ['user_id','item_id','rating','timestamp']
        
    elif size == "20m":
        df = pd.read_csv(filename,header=0,names=['user_id','item_id','rating','timestamp'])
    
    else:
        print("Not a proper size, or file not found")
        return
    
    # sort the dataframe by the timestamp, and drop the new "index" that appears from the sort
    df.sort_values('timestamp',inplace=True)
    
    # group all rows in the dataframe by user, and then get the length of each user session
    user_group = df.groupby('user_id').size()

    # get the number of unique users, items, ratings, and timestamps (number of unique values in each column)
    print(df.nunique())
    
    # get the shape of the dataframe (rows x columns)
    print(df.shape)
    
    # print statistics about the user session lengths such as max, min, and average
    print("Minimum Session Length: {:d}".format(user_group.min()))
    print("Maximum Session Length: {:d}".format(user_group.max()))
    print("Average Session Length: {:.2f}".format(user_group.mean()))
    return df.reset_index(drop=True)

def create_movie_df(filename=None,size="1m"):
    """
    Parameters
    ----------
    filename : string, optional
        The filename (and path) to the .csv file containing movie ids, title, genre, imdbid, tmdbid, plot . The default is None.

    Returns
    -------
    pandas dataframe
        returns a sorted by timestamp pandas dataframe with 4 columns: user id, item id, rating, and timestamp.
    """
    print("="*10,"Creating Movie Info DataFrame","="*10)
    
    
    if size == "1m":
        # read in the .dat file, and the entries on each line are separated by '::'
        df = pd.read_csv(filename,sep='::',header=None)
        df.columns= ["item_id", "title","genre","imdb_id","tmbd_id","mplot"]
        
    elif size == "20m":
           # read in the .csv file, and the entries
        df = pd.read_csv(filename,header=0,names=["item_id", "title","genre","imdb_id","tmbd_id","mplot"])
        
    
    # get the shape of the dataframe (rows x columns)
    print(df.shape)
    
    plot_sizes = df[-df.mplot.isna()].mplot.apply(lambda x: len(str(x).split()))
    number_missing = df.mplot.isna().sum()
    
    # if there is no movie plot for a movie, make the movie plot the title of the movie
    df.mplot[df.mplot.isna()] = df.title[df.mplot.isna()]
    
    # convert all the movie plots to strings (inacase there are just numbers)
    df.mplot = df.mplot.apply(str)
    
    # print statistics about the user session lengths such as max, min, and average
    print("Minimum Plot Length: {:d}".format(plot_sizes.min()))
    print("Maximum Plot Length: {:d}".format(plot_sizes.max()))
    print("Average Plot Length: {:.2f}".format(plot_sizes.mean()))
    
    print("Number of missing plots: {:d}".format(number_missing))
    
    # get the number of unique movie id, etc (number of unique values in each column)
    print(df.nunique())

    return df.reset_index(drop=True)


def filter_df(df=None,item_min=10):
    """
    Parameters
    ----------
    df : pandas dataframe, optional
        The pandas dataframe where each row is a user id, item id, rating, and timestamp. The default is None.
    item_min : TYPE, optional
        The minimum number of items required in a session length (otherwise get rid of user session). The default is 10.

    Returns
    -------
    filter_df : pandas dataframe
        The input dataframe, but now any user with a session length less than item_min is removed from the dataframe.

    """
    print("="*10,"Filtering Sessions <= {:d}  DataFrame".format(item_min),"="*10)

    if df is None:
        return 
    
    # groupo all the rows in the dataframe by user, and then get the length of each user session
    user_counts = df.groupby('user_id').size()
    
    # see which users have a session length greater than or equal to item_min
    user_subset = np.in1d(df.user_id,user_counts[user_counts >= item_min].index)
    
    # keep only the users with session length greater than or equal to item_min
    # reset the index...
    filter_df = df[user_subset].reset_index(drop=True)
    
    # check to make sure there are no user session lengths less than item_min
    assert (filter_df.groupby('user_id').size() < item_min).sum() == 0    
    
    # group all rows in the dataframe by user, and then get the length of each user session
    user_group = filter_df.groupby('user_id').size()
    
    # get the number of unique users, items, ratings, and timestamps (number of unique values in each column)
    print(filter_df.nunique())
    
    # get the shape of the dataframe (rows x columns)
    print(filter_df.shape)
    
    # print statistics about the user session lengths such as max, min, and average
    print("Minimum Session Length: {:d}".format(user_group.min()))
    print("Maximum Session Length: {:d}".format(user_group.max()))
    print("Average Session Length: {:.2f}".format(user_group.mean()))
    
    return filter_df



class reset_df(object):
    
    def __init__(self):
        print("="*10,"Initialize Reset DataFrame Object","="*10)
        
        # initialize labelencoder (which encode target labels with value between 0 and n_classes-1.)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()
        
    def fit_transform(self,df):
        """
        Parameters
        ----------
        df : pandas dataframe
            The pandas dataframe where each row is a user id, item id, rating, and timestamp. 

        Returns
        -------
        df : pandas dataframe
            The pandas dataframe with the item ids and user ids mapped to a value between 0 and n_unique_item_ids-1 and 0 and n_unique_user_ids-1 respectively.

        """
        print("="*10,"Resetting user ids and item ids in DataFrame","="*10)
        
        # encode item ids with value between 0 and n_classes-1.
        df['item_id'] = self.item_enc.fit_transform(df['item_id'])
        
        # encode movie ids with value between 0 and n_classes-1.
        df['user_id'] = self.user_enc.fit_transform(df['user_id'])
        
        # make sure that the item id and the user id both start at 0 
        assert df.user_id.min() == 0
        assert df.item_id.min() == 0 
        
        return df
    
    def inverse_transform(self,df):
        """
        Parameters
        ----------
        df : pandas dataframe
            The pandas dataframe where each row is a user id, item id, rating, and timestamp. 

        Returns
        -------
        df : pandas dataframe
            The pandas dataframe with the item ids and user ids mapped back to their original item ids and user ids respectively.
        """
        # Transform item ids back to original encoding.
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        
        # Transform user ids back to original encoding.
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df
    
def create_user_history(df=None):
    """
    Parameters
    ----------
    df : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    user_history : dictionary
        A dictionary where each key is the user id and each value is a list of the user session history.
        I.e. user_id = 5 , session = [1,4,7,2] ...

    """
    if df is None:
        return None
    
    print("="*10,"Creating User Histories","="*10)
    
    """
    # initialize empty user dictionary
    user_history = {}
    
    # iterate through each user id
    for uid in tqdm(df.user_id.unique()):
        # get the user session for user id uid
        sequence = df[df.user_id == uid].item_id.values.tolist()
        # save session as value in dictionary corresponding to key uid
        user_history[uid] = sequence
    """
        
    user_history = {uid : df[df.user_id == uid].item_id.values.tolist() for uid in tqdm(df.user_id.unique())}
    return user_history

def convert_genres(df, null_genre="NULL"):
    """
    Parameters
    ----------
    
    df: pandas dataframe
        dataframe that is outputed from create_movie_df
        
    Returns
    -------
    
    df: same df but with the genre column changed
    """
    new_df = df[['item_id', 'genre']].copy()
    new_df['genre'] = new_df.genre.apply(lambda x: x.split('|'))
    max_genres = new_df['genre'].apply(lambda x: len(x)).max()
    
    def filling_genres(x):
        fill_value = null_genre
        added_null = max_genres - len(x)
        return x + added_null * [fill_value]
    
    new_df['genre'] = new_df.genre.apply(filling_genres)
    return new_df

class reset_df_genres(object):
    
    def __init__(self):
        print("="*10,"Initialize Reset DataFrame Object","="*10)
        
        # initialize labelencoder (which encode target labels with value between 0 and n_classes-1.)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()
        self.genre_enc = LabelEncoder()
        
    def fit_transform(self, df, movie_df):
        """
        Parameters
        ----------
        df : pandas dataframe
            The pandas dataframe where each row is a user id, item id, rating, and timestamp. 
            
        movie_df: pandas dataframe
            dataframe that is outputed from convert_genres

        Returns
        -------
        df : pandas dataframe
            The pandas dataframe with the item ids and user ids mapped to a value between 0 and n_unique_item_ids-1 and 0 and n_unique_user_ids-1 respectively.
        
        movie_df
        """
        print("="*10,"Resetting user ids and item ids in DataFrame as well as movie DataFrame","="*10)
        
        new_movie_df = self.encoding_genres(movie_df)
        
        # encode item ids with value between 0 and n_classes-1.
        df['item_id'] = self.item_enc.transform(df['item_id'])
        
        # encode movie ids with value between 0 and n_classes-1.
        df['user_id'] = self.user_enc.fit_transform(df['user_id'])
        
        # make sure that the item id and the user id both start at 0 
        assert df.user_id.min() == 0
        assert df.item_id.min() == 0 
        
        return df, new_movie_df
    
    def encoding_genres(self, df):
        """
        Parameters
        ----------

        df: pandas dataframe
            dataframe that is outputed from convert_genres

        reset_encoders: encoder object used for the main dataframe

        Returns
        -------

        df: same df but with the genre and item column changed to encoding

        enc: the encoder for genres
        """
        new_df = df.copy()
        new_df['item_id'] = self.item_enc.fit_transform(df['item_id']) # this is problem id from movie excel sheet or id from ratings excel sheet. . .
        encodings = np.unique(np.concatenate(new_df['genre'].tolist())) 
        self.genre_enc.fit(encodings)
        new_df['genre'] = new_df.genre.apply(self.genre_enc.transform)
        
        return new_df
    
    def inverse_transform(self,df):
        """
        Parameters
        ----------
        df : pandas dataframe
            The pandas dataframe where each row is a user id, item id, rating, and timestamp. 

        Returns
        -------
        df : pandas dataframe
            The pandas dataframe with the item ids and user ids mapped back to their original item ids and user ids respectively.
        """
        # Transform item ids back to original encoding.
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        
        # Transform user ids back to original encoding.
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df

def train_val_test_split(user_history=None,max_length=200):
    """
    Leave-one-out training scheme
    Parameters
    ----------
    user_history : dictionary
        A dictionary where each key is the user id and each value is a list of the user session history.
        I.e. user_id = 5 , session = [1,4,7,2] ...
    max_length : integer, optional
        The maximum length that a user session is allowed to be. If a user session is longer than the maximum session length, it will be cut
        by taking the last max_length items. The default is 200.

    Returns
    -------
    train_history : dictionary
        A dictionary where each key is the user id and each value is a list of the last max_length items before the last 2 items in a user session history
        I.e. user_id = 5 , session = [1,4,7,2 ...] ....
    val_history : dictionary
        A dictionary where each key is the user id and each value is a list of the last max_length items before the last item in a user session history
        I.e. user_id = 5 , session = [1,4,7,2 ...] ....    test_history : dictionary
    test_history : dictionary
        A dictionary where each key is the user id and each value is a list of the last max_length items in a user session history
        I.e. user_id = 5 , session = [1,4,7,2 ...] ....    test_history : dictionary
    """
    if user_history is None:
        return None
    
    # add 1 to the maximum length of the parameter to truly get user sessions of max length
    # (is max length is 40, then we want 41 because 0:40 is train input whereas 1:41 is label)
    max_length = max_length + 1


    print("="*10,"Splitting User Histories into Train, Validation, and Test Splits","="*10)
    
    # initialize empty user dictionary for train, validation, and test dictionarys
    train_history = {}
    val_history = {}
    test_history = {}
    
    # iterate through each user and corresponding user session history
    for key,history in tqdm(user_history.items(),position=0, leave=True):
        
        # assign value for each key (uid) the the last max_length items before the last 2 items in a user session history
        train_history[key] = history[-(max_length+2):-2]
        # assign value for each key (uid) the last max_length items before the last item in a user session history
        val_history[key] = history[-(max_length+1):-1]
        # assign value for each key (uid) the last max_length items in a user session history
        test_history[key] = history[(-max_length):]
        
    return train_history,val_history,test_history

# TODO: make this better
def create_user_noclick(user_history,df,n_items):
    """
    Parameters
    ----------
    user_history : dictionary
        A dictionary where each key is the user id and each value is a list of the user session history.
        I.e. user_id = 5 , session = [1,4,7,2] ...
    df : pandas dataframe, optional
        The pandas dataframe where each row is a user id, item id, rating, and timestamp. The default is None.
    n_items : integer
        The number of unique items in the dataset.

    Returns
    -------
    user_noclick : dictionary
        A dictionary where the key is the user id and the value is a tuple of two lists.
        List 1 is a list of items the user has never clicked on.
        List 2 is a list of probabilities corresponding to popularity of items

    """
    print("="*10,"Creating User 'no-click' history","="*10)
    user_noclick = {}
    all_items = np.arange(n_items)

    # get the number of times each item id was clicked across all users
    item_counts = df.groupby('item_id',sort='item_id').size()
    #item_counts = (item_counts/item_counts.sum()).values


    # iterate through each user and corresponding user session history
    for uid,history in tqdm(user_history.items()):
        
        # get a list of all the items that user uid has no clicked on historically
        no_clicks = list(set.difference(set(all_items.tolist()),set(history)))
        
        # get the number of times each item id was clicked across all users for the subset
        item_counts_subset = item_counts[no_clicks]
        
        # normalize to get probabilities (more popular items have higher probability)
        probabilities = ( item_counts_subset/item_counts_subset.sum() ).values

        # assign the tuple of no click list and corresponding probabilities to the respective user id key
        user_noclick[uid] = (no_clicks,probabilities)
    
    return user_noclick