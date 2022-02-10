import numpy as np
import itertools

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
from pulp import GLPK

from sklearn.model_selection import train_test_split

import itertools
from ast import literal_eval


#from .estimators.GaussianKDE import *
#from .estimators.EmpiricalCovariance import *
    


def ISDE(X, m, n, k, multidimensional_estimator, **params_estimator):
    '''
    
    Inputs
    - X : input dataset (numpy array)
    - m : size of set use to 
    - n : size of set used ton compute log-likelihoods
    - k : desired size of biggest bloc in the outputed partition
    '''
    
    by_subsets = {}
    
    N, d = X.shape
    W, Z = train_test_split(X, train_size=m, test_size=n)
    
    for i in range(1, k+1):
        for S in itertools.combinations(range(d), i):
        
            f, f_params = multidimensional_estimator(W[:, S], params_estimator)
            ll = np.mean( f.score_samples(grid_points = W[:, S], eval_points = Z[:, S]) )
            
            by_subsets[S] = {'log_likelihood': ll, 'params': f_params}


            
    optimal_partition = find_optimal_partition(by_subsets, max_size=k, min_size=1, exclude = [], sense='maximize')[0]
    
    optimal_parameters = []
    for S in optimal_partition:
        optimal_parameters.append(by_subsets[tuple(S)]["params"])
        
    return by_subsets, optimal_partition, optimal_parameters




def find_optimal_partition(scores_by_subsets, max_size, min_size=1, exclude = [], sense='maximize'):
    
    nb_to_exclude = len(exclude)
    
    ### Create Graph
    weights = {}
    edges = []
    vertices = []

    for s in scores_by_subsets.keys():
        
            for i in s:
                if i not in vertices:
                    vertices.append(i)

            if len(s) <= max_size and len(s) >= min_size:
                edges.append(s)
                weights[s] = scores_by_subsets[s]["log_likelihood"]
    
    ### Create model and variables
    if sense == 'maximize':
        model = LpProblem(name="Best_partition", sense=LpMaximize)
    elif sense == 'minimize':
        model = LpProblem(name="Best_partition", sense=LpMinimize)
    xs = []
    
    for e in edges:
        #Replace ' ' by '' to avoid extras '_'
        xs.append(LpVariable(name=str(e).replace(' ', ''), lowBound=0, upBound=1, cat="Integer"))
    
    ### Cost function
    objective = lpSum([weights[e] * xs[i] for (i, e) in enumerate(edges)])
    model += objective
    

    ### Constrains
    A = np.zeros(shape=(len(vertices), len(edges)))
    for (i, e) in enumerate(edges):
        for v in e:
            A[v, i] = 1
    
    for (i, e) in enumerate(vertices):
        model += (lpSum([A[i, j] * xs[j] for j in range(len(edges)) ]) == 1)
        
    
    ### exclude
    if len(exclude) > 1:
    
        xs_name = [ list(literal_eval(i.name)) for i in xs]
        for p_exclude in exclude:
            model += lpSum( [xs[xs_name.index(s)] for s in p_exclude]) <= len(p_exclude) - 1 
    
    #Solve
    model.solve()
    #if verbose != 0:
        #print("Status: {}, {}".format(model.status, LpStatus[model.status]))
        #print("Objective value : {}".format(model.objective.value()))
        
    output_dict = {var.name : var.value() for var in model.variables()  }
    out_partition = []
    for o in output_dict.keys():
        if output_dict[o] != 0:
            out_partition.append(list(literal_eval(o.replace("_", " "))))
            
    #if verbose != 0:
        #print("Output : {}".format(out_partition))
    
    return out_partition, model.objective.value()

        
def logdensity_from_partition(X, X_eval, partition, parameters, estimator):
    
    M = len(X_eval)
    log_density = np.zeros(len(X_eval))

    for i, S in enumerate(partition):

        loc_param = parameters[i]
        f = estimator(**loc_param)
        log_density += f.score_samples(grid_points=X[:, S], eval_points=X_eval[:, S])

    return log_density







