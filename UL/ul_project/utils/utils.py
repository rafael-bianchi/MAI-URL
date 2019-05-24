import numpy as np
import os

def read_dataset(dataset_name):
    '''
    Reads the dataset into a numpy array

    Keyword arguments:
    dataset_name -- The dataset name. 

    Outputs:
    X -- dataset
    Y -- cluster labels
    N -- Number of samples
    d -- number of features (dimension)
    '''
    root_dataset_folder = 'ul_project/datasets'
    filename = os.path.join(root_dataset_folder, f'{dataset_name}.csv')

    dataset = np.genfromtxt(filename, delimiter=',')

    X = dataset[:, :-1]
    Y = dataset[:, -1]
    N, d = X.shape

    return X, Y, N, d

def get_initial_clusters(X, k):
    '''
    Randomly initialize the cluster centers.

    Keyword arguments:
    X -- Numpy dataset.
    k -- number of clusters 

    Outputs:
    M -- Initial cluster [k,:]
    '''
    M = X[np.random.choice(X.shape[0], k, replace=False), :]
    return M