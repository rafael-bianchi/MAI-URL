import math
import os
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler


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
    filename = os.path.join(root_dataset_folder, f'{dataset_name}')

    ext = Path(filename).suffix

    dataset = None
    X = Y = None

    if (ext == '.csv'):
        dataset = np.genfromtxt(filename, delimiter=',')
        X = dataset[:, :-1]
        Y = dataset[:, -1]
    else:
        X, Y = read_arff(filename)

    N, d = X.shape

    return X, Y, N, d, len(np.unique(Y))

def read_arff(fileName):
    raw_data, meta = loadarff(fileName)
    x, y = pre_process_data(raw_data, meta)

    return x, y

def pre_process_data(raw_data, meta):
    data = pd.DataFrame()

    feature_names = meta.names()
    for col_name, type_info in [(i, meta[i]) for i in feature_names[:-1]]:
        if type_info[0] == "nominal":
            if encode:
                if (b'?' in raw_data[col_name]):
                    val_mode = statistics.mode([val for val in raw_data[col_name] if val != b'?'])
                    raw_data[col_name] = [val_mode if val == b'?' else val for val in raw_data[col_name]]

                one_hot_encoded_data = one_hot_encode(
                    [self._process_byte_string(val) for val in raw_data[col_name]],
                    col_name
                )
                data = pd.concat([data, one_hot_encoded_data], axis=1, sort=False)
            else:
                data = pd.concat([
                    data, 
                    pd.Series(
                        [_process_byte_string(val) for val in raw_data[col_name]],
                        name=col_name
                    ).astype('category')
                ], axis=1, sort=False)

        if type_info[0] == "numeric":
            mean = statistics.mean([val for val in raw_data[col_name] if not math.isnan(val)])
            data[col_name]= [mean if math.isnan(val) else val for val in raw_data[col_name]]

    target_class_name = feature_names[-1]
    target_class = pd.Series(
        [_process_byte_string(val) for val in raw_data[target_class_name]],
        name=target_class_name
    ).astype('category').cat.codes

    scaled_values = MinMaxScaler().fit_transform(data.values.astype(float))
    normalized = pd.DataFrame(data=scaled_values, columns=data.columns)
    return normalized.to_numpy(), target_class.to_numpy()

def _process_byte_string(bstring):
    return bstring.decode("utf-8").strip("'")

def one_hot_encode(data, prefix=""):
    return pd.get_dummies(data, prefix=prefix)

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
