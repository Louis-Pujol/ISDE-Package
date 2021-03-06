import numpy as np
from pykeops.numpy import LazyTensor as LazyTensor_np

from pykeops.numpy import Genred

from time import time
import os





def gaussian_kde(grid_points, eval_points, h, backend='auto'):
    

    N, d = grid_points.shape
    
    my_conv = Genred('Exp(- SqNorm2(x - y))', ['x = Vi({})'.format(d), 'y = Vj({})'.format(d)],
                     reduction_op='Sum',axis=0)

    
    C = np.sqrt(0.5) / h
    a = my_conv(C * np.ascontiguousarray(grid_points)
                ,C * np.ascontiguousarray(eval_points),
                backend=backend).transpose()[0]
    
    return a / (N*(h ** d)*np.power(2*np.pi, d/2))


def gaussian_kde_old(grid_points, eval_points, h):
    """
    Perform Kernel Density estimation with grid_points and eval_points and a vector of bandwidths
    Works with gpu and torch (and cpu ?)
    
    Inputs : 
    - eval_points : (m, d) array, points to evaluate density
    - grid_points : (N, d), points to construct density estimator
    - h : d-dimensional array, directional bandwidths
    
    Output:
    - m-dimensional array, evaluation of density for points of eval_points
    """
    
    N, d = grid_points.shape
    x_i = LazyTensor_np(np.ascontiguousarray(eval_points[:, None, :]))  # (M, 1, d) KeOps LazyTensor, wrapped around the numpy array eval_points
    X_j = LazyTensor_np(np.ascontiguousarray(grid_points[None, :, :]))  # (1, N, d) KeOps LazyTensor, wrapped around the numpy array grid_points
    h_l = LazyTensor_np(np.ascontiguousarray(h))

    D_ij = ( -0.5 * (((x_i - X_j) / h_l) ** 2).sum(-1))  # **Symbolic** (M, N) matrix of squared distances
    s_i = D_ij.exp().sum(dim=1).ravel()  # genuine (M,) array of integer indices
    
    out = s_i / (N*np.prod(h)*np.power(2*np.pi, d/2))
    
    return out




class GaussianKDE:
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def score_samples(self, grid_points, eval_points):
        """ Return log-likelihood evaluation
        """
        _, d = eval_points.shape
        return np.log( gaussian_kde(grid_points=grid_points, eval_points=eval_points, h=self.bandwidth) )
    
def CVKDE(W, params):
    
    hs = params['hs']
    if 'n_fold' in params:
        n_fold = params['n_fold']
    else:
        n_fold = 5
    
    
    scores = {h : [] for h in hs}
    
    m, d = W.shape
    step = int(m / n_fold)
    indexes = list(range(m))

    for i in range(int(m/step)):
                            
        if i != int(m/step) - 1 :
            test_indexes = indexes[i * step: (i+1)*step]
            train_indexes = [i for i in indexes if i not in test_indexes]
                                
        else:
            test_indexes = indexes[i * step:]
            train_indexes = [i for i in indexes if i not in test_indexes]
                                
        W_train, W_test = W[train_indexes, :], W[test_indexes, :]
                            
        for h in hs:
            scores[h].append(np.mean (np.log( gaussian_kde(grid_points=W_train, eval_points=W_test, h=h))))
            
    mean_scores = [np.mean(scores[h]) for h in hs]
    h_opt = hs[np.argmax(mean_scores)]
    
    kde = GaussianKDE(bandwidth=h_opt)
        
    return kde, {'bandwidth' : h_opt}

def Hold_out_KDE(W,params):
    
    hs = params['hs']
    n_train = params['n_train']
    
    W_train = W[0:n_train, :] 
    W_test = W[n_train::, ]
    
    scores = {h : [] for h in hs}
    for h in hs:
        tmp = np.log(gaussian_kde(grid_points=W_train, eval_points=W_test, h=h))
        scores[h].append(np.ma.masked_invalid(tmp).mean())
    
    
    h_opt = hs[np.argmax([scores[h] for h in hs])]
    kde = GaussianKDE(bandwidth=h_opt)
        
    return kde, {'bandwidth' : h_opt}

    
def KDE_fixed_h(W, params):
    
    h = params['h']
    
    if h == 'scott':
        n, d = W.shape
        h = n ** (- 1. / (d + 4))
    
    kde = GaussianKDE(bandwidth=h)
        
    return kde, {'bandwidth' : h}
