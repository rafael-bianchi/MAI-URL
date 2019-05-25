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

def get_initial_clusters(X, k, iter=0):
    '''
    Randomly initialize the cluster centers.

    Keyword arguments:
    X -- Numpy dataset.
    k -- number of clusters 

    Outputs:
    M -- Initial cluster [k,:]
    '''

    seed = []
    seed.append((113, 116, 165))
    seed.append((26, 195, 165))
    seed.append((124, 21, 68))
    seed.append((88, 60, 9))
    seed.append((108, 186, 74))
    seed.append((152, 195, 119))
    seed.append((94, 77, 41))
    seed.append((13, 71, 105))
    seed.append((49, 167, 201))
    seed.append((5, 33, 66))

    #M = X[np.random.choice(X.shape[0], k, replace=False), :]
    M = X[seed[iter], :]
    return M