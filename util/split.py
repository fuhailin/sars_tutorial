import random

from scipy.sparse import find
from sklearn.model_selection import train_test_split


def random_holdout(dataset, perc=0.8, seed=1234):
    """
    Split sequence dataset randomly
    :param dataset: the sequence dataset
    :param perc: the training percentange
    :param seed: the random seed
    :return: the training and test splits
    """
    dataset = dataset.sample(frac=1, random_state=seed)
    nseqs = len(dataset)
    train_size = int(nseqs * perc)
    # split data according to the shuffled index and the holdout size
    train_split = dataset[:train_size]
    test_split = dataset[train_size:]

    return train_split, test_split


def temporal_holdout(dataset, ts_threshold):
    """
    Split sequence dataset using timestamps
    :param dataset: the sequence dataset
    :param ts_threshold: the timestamp from which test sequences will start
    :return: the training and test splits
    """
    train_split = dataset.loc[dataset['ts'] < ts_threshold]
    test_split = dataset.loc[dataset['ts'] >= ts_threshold]

    return train_split, test_split


def last_session_out_split(data,
                           user_key='UserID',
                           session_key='session_id',
                           time_key='ts'):
    """
    Assign the last session of every user to the test set and the remaining ones to the training set
    """
    # train = data.copy()
    # test = data.copy()
    #
    # test['sequence'] = test['sequence'].apply(lambda x: x[-1])
    # train['sequence'] = train['sequence'].apply(lambda x: x[:-1])
    train, test = train_test_split(data, test_size=0.1, random_state=12)

    # sessions = data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
    # last_session = sessions.last()
    # train = data[~data['UserID'].isin(last_session.values)].copy()
    # test = data[data['UserID'].isin(last_session.values)].copy()
    return train, test


def balance_dataset(x, y):
    number_of_elements = y.shape[0]
    nnz = set(find(y)[0])
    zero = set(range(number_of_elements)).difference(nnz)

    max_samples = min(len(zero), len(nnz))

    nnz_indices = random.sample(nnz, max_samples)
    zero_indeces = random.sample(zero, max_samples)
    indeces = nnz_indices + zero_indeces

    return x[indeces, :], y[indeces, :]
