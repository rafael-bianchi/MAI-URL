
import math
import numpy as np
import scipy.spatial.distance as dist

def MinMax_kmeans(X,M,k,p_init,p_max,p_step,t_max,beta):
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
    Cluster_elem -- N-dimensional row vector containing the final cluster assignments.
    Clusters -- indexed 1,2,...,k.
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
            print('p_step reset to zero, since p_max equals p_init\n\n')
        p_flag = 0
        p_step = 0    
    elif p_step==0:
        if p_init!=p_max:
            print('p_max reset to equal p_init, since p_step=0\n\n')
        
        p_flag=0
        p_max=p_init        
        
    else:
        p_flag=1 #p_flag indicates whether p will be increased during the iterations.

    pass
    # #--------------------------------------------------------------------------
    # #Weights are uniformly initialized.
    W=np.ones(shape=k)/k 

    #Other initializations.
    p = p_init #Initial p value.
    p_prev = p-10^(-8) #Dummy value.
    empty = 0 #Count the number of iterations for which an empty or singleton cluster is detected.
    Iter = 1 #Number of iterations.
    E_w_old = math.inf #Previous iteration objective (used to check convergence).
    Var = np.zeros(k)
    Cluster_elem_history = []
    W_history = []

    #--------------------------------------------------------------------------
    print('\nStart of MinMax k-means iterations\n')
    print('----------------------------------\n\n')

    #Calculate the squared Euclidean distances between the instances and the
    #initial cluster centers.    
    Dist = dist.cdist(X, M, 'euclidean')

    #The MinMax k-means iterative procedure.
    while (True):
        #Calculate the weighted distances.
        #Each cluster is multipied by its weight.
        WDist=Dist
        for i in range(0,k):
            WDist[:,i] = W[i]**(p*Dist[:,i])

        #Update the cluster assignments.
        [min_WDist,Cluster_elem]=min(WDist,[],1)
        
        # #Calculate the MinMax k-means objective.
        # E_w=sum(min_WDist)    
        
        # #If empty or singleton clusters are detected after the update.
        # for i in range(1,k): 
        #     I=find(Cluster_elem==i)
        #     if isempty(I) or length(I)==1:
                        
        #         print('Empty or singleton clusters detected for p=%g.\n',p)
        #         print('Reverting to previous p value.\n\n')           
                
        #         E_w=NaN #Current objective undefined.
        #         empty=empty+1
                
        #         #Reduce p when empty or singleton clusters are detected.
        #         if empty > 1:
        #             p=p-p_step

        #         #The last p increase may not correspond to a complete p_step,
        #         #if the difference p_max-p_init is not an exact multiple of p_step.
        #         else:
        #             p=p_prev
                
        #         p_flag=0 #Never increase p again.
                                    
        #         #p is not allowed to drop out of the given range.
        #         if p<p_init or p_step==0:
                    
        #             print('\n+++++++++++++++++++++++++++++++++++++++++\n\n')
        #             print('p cannot be decreased further.\n')
        #             print('Either p_step=0 or p_init already reached.\n')
        #             print('Aborting Execution\n')
        #             print('\n+++++++++++++++++++++++++++++++++++++++++\n\n')
                    
        #             #Return NaN to indicate that no solution is produced.
        #             Cluster_elem=NaN(1,size(X,1))
        #             M=NaN(k,size(X,2))
        #             Var=NaN(1,k)

        #             return Cluster_elem, M, Var
                
        #         #Continue from the assignments and the weights corresponding 
        #         #to the decreased p value.
        #         Cluster_elem=Cluster_elem_history(empty,:)
        #         W=W_history(empty,:)
        #         break
        
        # if not math.isnan(E_w):
        #     print('p=%g\n',p)
        #     print('The MinMax k-means objective is E_w=%f\n\n',E_w)
                    
        # #Check for convergence. Never converge if in the current (or previous)
        # #iteration empty or singleton clusters were detected.
        # if ~isnan(E_w) and ~isnan(E_w_old) and (abs(1-E_w/E_w_old)<1e-6 or Iter>=t_max): 
            
        #     #Calculate the cluster variances.
        #     for i=1:k
        #         I=Cluster_elem==i
        #         Var(i)=sum(Dist(i,I))
        #     end
            
        #     print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
        #     print('Converging for p=%g after %d iterations.\n',p,Iter)               
        #     print('The final MinMax k-means objective is E_w=%f.\n',E_w)       
        #     print('The maximum cluster variance is E_max=%f.\n',max(Var))
        #     print('The sum of the cluster variances is E_sum=%f.\n',sum(Var))
        #     print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
            
        #     break
                            
        # E_w_old = E_w
        
        # #Update the cluster centers.
        # for i=1:k:
        #     I=Cluster_elem==i
        #     M(i,:)=mean(X(I,:),1)
            
        # #Increase the p value.
        # if p_flag==1
        #     #Keep the assignments-weights corresponding to the current p.
        #     #These are needed when empty or singleton clusters are found in
        #     #subsequent iterations.
        #     Cluster_elem_history=[Cluster_elemCluster_elem_history]
        #     W_history=[WW_history]
            
        #     p_prev=p
        #     p=p+p_step

        #     if p>=p_max:
        #         p=p_max
        #         p_flag=0
        #         print('p_max reached\n\n')
            
        # #Recalculate the distances between the instances and the cluster centers.
        # Dist=sqdist(M',X')
    
        # #Calculate the cluster variances.
        # for i=1:k
        #     I=Cluster_elem==i
        #     Var(i)=sum(Dist(i,I))
        
        # W_old=W
        
        # #Update the weights.
        # for i=1:k:
        #     W(i)=0
            
        #     for j=1:k:
        #         W(i)=W(i)+(Var(i)/Var(j)).^(1/(p-1))
            
        #     W(i)=1/W(i)
        
        # #Memory effect.
        # W=(1-beta)*W+beta*W_old
        
        # Iter=Iter+1