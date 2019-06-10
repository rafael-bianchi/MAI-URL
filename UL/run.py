import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

import ul_project.model.minmax_kmeans as mk
import ul_project.utils.utils as utils

X, Y, N, d, k_read = utils.read_dataset('Coil2.csv')

# MinMax Algorithm parameters.
# ---------------------
k=k_read#number of clusters.
p_init=0        #initial p.
p_max=0.5       #maximum p.
p_step=0.01     #p step.
t_max=500       #maximum number of iterations.
beta=0.0        #amount of memory for the weight updates.
Restarts=10     #number of MinMax k-means restarts.
# ---------------------


# Cluster the instances using the MinMax k-means procedure.
# ---------------------------------------------------------
E_max=np.zeros(shape = (Restarts, 2), dtype = 'float')
E_sum=np.zeros(shape = (Restarts, 2), dtype = 'float')
nmi_avg = np.zeros(shape = (Restarts, 2), dtype = 'float')

debug = False

for repeat in range(0, Restarts):
    if debug:
        print('========================================================\n')
        print(f'MinMax k-means: Restart {repeat+1}\n')
    
    #Randomly initialize the cluster centers.
    M = utils.get_initial_clusters(X, k, repeat)
    
    #Execute MinMax k-means.
    #Get the cluster assignments, the cluster centers and the cluster variances.
    Cluster_elem, M, Var_MinMax = mk.MinMax_kmeans(X, M, k, p_init, p_max, p_step, t_max, beta, debug=debug)
    
    E_max[repeat, 0]=max(Var_MinMax)
    E_sum[repeat, 0]=sum(Var_MinMax)
    nmi_avg[repeat, 0] = normalized_mutual_info_score(Y, Cluster_elem, average_method='arithmetic')

    #K-means
    kmeans = KMeans(n_clusters=k, init=M, max_iter=t_max, n_init=1).fit(X)
    Dist = dist.cdist(X, kmeans.cluster_centers_, 'sqeuclidean')
    #Calculate the cluster variances.
    Var = np.zeros(k)
    for i in range(0,k):
        I = np.where(kmeans.labels_ == i)
        Var[i] = np.sum(Dist[I[0],i])

    E_max[repeat, 1]=max(Var)
    E_sum[repeat, 1]=sum(Var)
    nmi_avg[repeat, 1] = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')


    if debug:
        print(f'End of Restart {repeat+1}\n')
        print('========================================================\n')

print(f'Average E_max score over {Restarts+1} restarts: {E_max[:,0].mean()}.')
print(f'Average E_sum score over {Restarts+1} restarts: {E_sum[:,0].mean()}.')
print(f'Average nmi score over {Restarts+1} restarts: {nmi_avg[:,0].mean()}.')

# ---------------------------------------------------------
