import ul_project.utils.utils as utils
import ul_project.model.minmax_kmeans as mk
import numpy as np

X, Y, N, d = utils.read_dataset('Coil2')

# Algorithm parameters.
# ---------------------
k=3#number of clusters.
p_init=.5        #initial p.
p_max=0.5       #maximum p.
p_step=0.01     #p step.
t_max=500       #maximum number of iterations.
beta=0.3        #amount of memory for the weight updates.
Restarts=10     #number of MinMax k-means restarts.
# ---------------------

# Cluster the instances using the MinMax k-means procedure.
# ---------------------------------------------------------
E_max=np.zeros(shape = Restarts, dtype = 'float')
E_sum=np.zeros(shape = Restarts, dtype = 'float')

for repeat in range(0, Restarts):
    print('========================================================\n')
    print(f'MinMax k-means: Restart {repeat+1}\n')
    
    #Randomly initialize the cluster centers.
    M = utils.get_initial_clusters(X, k, repeat)
    
    #Execute MinMax k-means.
    #Get the cluster assignments, the cluster centers and the cluster variances.
    Cluster_elem, M, Var = mk.MinMax_kmeans(X, M, k, p_init, p_max, p_step, t_max,beta)
    
    E_max[repeat]=max(Var)
    E_sum[repeat]=sum(Var)

    print(f'End of Restart {repeat+1}\n')
    print('========================================================\n')

print(f'Average E_max score over {Restarts+1} restarts: {E_max.mean()}.')
print(f'Average E_sum score over {Restarts+1} restarts: {E_sum.mean()}.')

# ---------------------------------------------------------
