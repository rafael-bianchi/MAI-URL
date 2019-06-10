import math

import numpy as np
import scipy.spatial.distance as dist

from sklearn.metrics.pairwise import euclidean_distances

def MinMax_kmeans(X, M, k, p_init, p_max, p_step, t_max, beta, debug = True):
    """ 
    This function implements the MinMax k-means algorithm as described in  
    G.Tzortzis and A.Likas, "The MinMax k-means Clustering Algorithm", Pattern Recognition, 2014.

    Keyword arguments:
    X -- Nxd data matrix, where each row corresponds to an instance.
    M -- kxd matrix of the initial cluster centers. Each row corresponds to a center.
    k -- the number of clusters.
    p_init -- initial p value (0<=p_init<1).
    p_max -- maximum admissible p value (0<=p_max<1).
    p_step -- step for increasing p (p_step>=0).
    t_max -- maximum number of iterations (necessary as convergence cannot be guaranteed for MinMax k-means).
    beta -- controls the amount of memory for the weight updates (0<=beta<=1).
    
    Outputs:
    Cluster_elem -- N-dimensional row vector containing the final cluster assignments. Clusters indexed 1,2,...,k.
    M -- kxd matrix of the final cluster centers. Each row corresponds to a center.
    Var -- k-dimensional row vector containing the final variance of each cluster.
    """

    if p_init < 0 or p_init >= 1:
        raise('p_init must take a value in [0,1)')

    if p_max < 0 or p_max >= 1:
        raise('p_max must take a value in [0,1)')

    if p_max < p_init:
        raise('p_max must be greater or equal to p_init')

    if p_step < 0:
        raise('p_step must be a non-negative number')

    if beta < 0 or beta > 1:
        raise('beta must take a value in [0,1]')

    if p_init == p_max:
        if p_step != 0:
            print('p_step reset to zero, since p_max equals p_init\n')
        p_flag = 0
        p_step = 0    
    elif p_step==0:
        if p_init!=p_max:
            print('p_max reset to equal p_init, since p_step=0\n')
        
        p_flag=0
        p_max=p_init        
        
    else:
        p_flag=1 #p_flag indicates whether p will be increased during the iterations.

    # #--------------------------------------------------------------------------
    # #Weights are uniformly initialized.
    W=np.ones(shape=k)/k 

    #Other initializations.
    p = p_init #Initial p value.
    p_prev = p-10**(-8) #Dummy value.
    empty = 0 #Count the number of iterations for which an empty or singleton cluster is detected.
    Iter = 0 #Number of iterations.
    E_w_old = math.inf #Previous iteration objective (used to check convergence).
    Var = np.zeros(k)
    Cluster_elem_history = []
    W_history = []

    #--------------------------------------------------------------------------
    if debug:
        print('\nStart of MinMax k-means iterations')
        print('----------------------------------\n')

    #Calculate the squared Euclidean distances between the instances and the
    #initial cluster centers.    
    Dist = dist.cdist(X, M, 'sqeuclidean')

    #The MinMax k-means iterative procedure.
    while (True):
        #Calculate the weighted distances.
        #Each cluster is multipied by its weight.
        WDist=Dist
        for i in range(0,k):
            WDist[:,i] = W[i]**p*Dist[:,i]

        #Update the cluster assignments.
        Cluster_elem = np.argmin(WDist, axis=1)
        min_WDist = np.min(WDist, axis=1)
        
        # #Calculate the MinMax k-means objective.
        E_w = np.sum(min_WDist)
        
        #If empty or singleton clusters are detected after the update.
        for i in range(0,k): 
            I = np.where(Cluster_elem == i)
            if I[0].size <= 1:
                if debug:
                    print(f'Empty or singleton clusters detected for p={p}.')
                    print('Reverting to previous p value.\n')           
                
                E_w = math.nan #Current objective undefined.
                empty = empty+1
                
                #Reduce p when empty or singleton clusters are detected.
                if empty > 1:
                    p=p-p_step

                #The last p increase may not correspond to a complete p_step,
                #if the difference p_max-p_init is not an exact multiple of p_step.
                else:
                    p = p_prev
                
                p_flag = 0 #Never increase p again.
                                    
                #p is not allowed to drop out of the given range.
                if p < p_init or p_step == 0:    
                    if debug:                
                        print('+++++++++++++++++++++++++++++++++++++++++\n')
                        print('p cannot be decreased further.')
                        print('Either p_step=0 or p_init already reached.')
                        print('Aborting Execution\n')
                        print('+++++++++++++++++++++++++++++++++++++++++\n')
                    
                    #Return NaN to indicate that no solution is produced.
                    Cluster_elem.fill(np.nan)
                    M.fill(np.nan)
                    Var.fill(np.nan)

                    return Cluster_elem, M, Var
                
                #Continue from the assignments and the weights corresponding 
                #to the decreased p value.
                Cluster_elem=Cluster_elem_history[empty-1]
                W=W_history[empty-1]
                break
        
        if not math.isnan(E_w) and debug:
            print(f'p={p}')
            print(f'The MinMax k-means objective is E_w={E_w}\n')
                    
        # #Check for convergence. Never converge if in the current (or previous)
        # #iteration empty or singleton clusters were detected.
        if not math.isnan(E_w) and not math.isnan(E_w_old) and (abs(1-E_w/E_w_old) < 1e-6 or Iter >= t_max): 
            #Calculate the cluster variances.
            for i in range(0, k):
                I = np.where(Cluster_elem == i)
                Var[i] = np.sum(Dist[I,i])
            
            if debug:
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
                print(f'Converging for p={p} after {Iter} iterations.')               
                print(f'The final MinMax k-means objective is E_w={E_w}.')       
                print(f'The maximum cluster variance is E_max={max(Var)}.')
                print(f'The sum of the cluster variances is E_sum={sum(Var)}.')
                print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

            return Cluster_elem, M, Var

        E_w_old = E_w
        
        #Update the cluster centers.
        for i in range(0, k):
            I = np.where(Cluster_elem == i)
            M[i,:]= X[I[0],:].mean(axis=0)
            
        # #Increase the p value.
        if p_flag == 1:
            #Keep the assignments-weights corresponding to the current p.
            #These are needed when empty or singleton clusters are found in
            #subsequent iterations.
            Cluster_elem_history.append(Cluster_elem)  #=[Cluster_elem;Cluster_elem_history];
            W_history.append(W) #= [W;W_history];
            
            p_prev = p
            p = p + p_step

            if p >= p_max:
                p = p_max
                p_flag = 0
               
                if debug:
                    print('p_max reached\n')
            
        # #Recalculate the distances between the instances and the cluster centers.
        Dist = dist.cdist(X, M, 'sqeuclidean')
    
        #Calculate the cluster variances.
        for i in range(0,k):
            I = np.where(Cluster_elem == i)
            Var[i] = np.sum(Dist[I[0],i])
        
        W_old = np.copy(W)
        
        #Update the weights.
        for i in range(0,k):
            W[i] = 0
            
            for j in range(0,k):
                W[i]=W[i] + (Var[i]/Var[j])**(1/(p-1))
            
            W[i] = 1/W[i]
        
        #Memory effect.
        W = (1-beta) * W + beta * W_old
        
        Iter=Iter+1
