#%% Imports
import time

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from tabulate import tabulate

import ul_project.model.minmax_kmeans as mk
import ul_project.utils.utils as utils

#%%
# MinMax Algorithm parameters.
# ---------------------
p_init=0        #initial p.
p_max=0.5       #maximum p.
p_step=0.1     #p step.
t_max=500       #maximum number of iterations.
#beta=0.0        #amount of memory for the weight updates.
Restarts=500     #number of MinMax k-means restarts.
# ---------------------

debug = False
algorithms = ['MinMax KMeans (β = 0.0)', 'MinMax KMeans (β = 0.1)', 'MinMax KMeans (β = 0.3)', 'KMeans', 'KMeans++']

for dataset in ['blob']:#['blob', 'pen-based.arff', 'Coil2.csv', 'iris.arff']:
    #%%
    X, Y, N, d, k = utils.read_dataset(dataset)

    #%%
    # Cluster the instances using the MinMax k-means procedure.
    # ---------------------------------------------------------
    E_max=np.zeros(shape = (Restarts, len(algorithms)), dtype = 'float')
    E_sum=np.zeros(shape = (Restarts, len(algorithms)), dtype = 'float')
    nmi_avg = np.zeros(shape = (Restarts, len(algorithms)), dtype = 'float')
    time_avg = np.zeros(shape = (Restarts, len(algorithms)), dtype = 'float')

    for repeat in range(0, Restarts):
        alg_count = 0

        if debug:
            print('========================================================\n')
            print(f'Running all algorithms: Restart {repeat+1}\n')
        
        #Randomly initialize the cluster centers.
        M = utils.get_initial_clusters(X, k, repeat)
    
        for beta in [0.0, 0.1, 0.3]:
            M_temp = np.copy(M)
            #Execute MinMax k-means.
            #Get the cluster assignments, the cluster centers and the cluster variances.
            start = time.time()
            Cluster_elem, _, Var_MinMax = mk.MinMax_kmeans(X, M_temp, k, p_init, p_max, p_step, t_max, beta, debug=debug)
            end = time.time()

            nan_found = np.any(np.isnan(Var_MinMax))
                
            E_max[repeat, alg_count] = max(Var_MinMax)
            E_sum[repeat, alg_count] = sum(Var_MinMax)
            nmi_avg[repeat, alg_count] = normalized_mutual_info_score(Y, Cluster_elem, average_method='arithmetic') if not nan_found else np.nan
            time_avg[repeat, alg_count] = end - start if not nan_found else np.nan

            alg_count += 1

        #K-means
        M_temp = np.copy(M)
        start = time.time()
        kmeans = KMeans(n_clusters=k, init=M_temp, max_iter=t_max, n_init=1).fit(X)
        end = time.time()

        dist_kmeans = dist.cdist(X, kmeans.cluster_centers_, 'sqeuclidean')

        #Calculate the cluster variances.
        var_kmeans = np.zeros(k)
        for i in range(0,k):
            I = np.where(kmeans.labels_ == i)
            var_kmeans[i] = np.sum(dist_kmeans[I[0],i])

        E_max[repeat, alg_count]=max(var_kmeans)
        E_sum[repeat, alg_count]=sum(var_kmeans)
        nmi_avg[repeat, alg_count] = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
        time_avg[repeat, alg_count] = end - start

        alg_count += 1


        #K-means++
        start = time.time()
        kmeans_plusplus = KMeans(n_clusters=k, init='k-means++', max_iter=t_max, n_init=1).fit(X)
        end = time.time()

        dist_kmeans_plusplus = dist.cdist(X, kmeans_plusplus.cluster_centers_, 'sqeuclidean')

        #Calculate the cluster variances.
        var_kmeans_plusplus = np.zeros(k)
        for i in range(0,k):
            I = np.where(kmeans_plusplus.labels_ == i)
            var_kmeans_plusplus[i] = np.sum(dist_kmeans_plusplus[I[0],i])

        E_max[repeat, alg_count]=max(var_kmeans_plusplus)
        E_sum[repeat, alg_count]=sum(var_kmeans_plusplus)
        nmi_avg[repeat, alg_count] = normalized_mutual_info_score(Y, kmeans_plusplus.labels_, average_method='arithmetic')
        time_avg[repeat, alg_count] = end - start

        alg_count += 1

        if debug:
            print(f'Running all algorithms {repeat+1}\n')
            print('========================================================\n')

    #%%
    # Summarizing results 
    summary = pd.DataFrame({'Algorithm': algorithms})
    summary = summary.assign(e_max = np.nanmedian(E_max, axis=0), e_sum = np.nanmedian(E_sum, axis=0), nmi = np.nanmedian(nmi_avg, axis=0), time = np.nanmedian(time_avg, axis=0))

    print(f'Results for dataset {dataset}')
    print(tabulate(summary, headers='keys', tablefmt='latex_booktabs'))
