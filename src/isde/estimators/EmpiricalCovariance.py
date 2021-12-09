import numpy as np
from sklearn.covariance import empirical_covariance
from scipy.stats import multivariate_normal

class Covariance:
    
    def __init__(self, cov):
        self.cov = cov
        
        
    
    def score_samples(self, grid_points, eval_points):
        """ Return log-likelihood evaluation
        """
        _, d = eval_points.shape
        var = multivariate_normal(mean=np.zeros(d), cov=self.cov)
        return np.log(var.pdf(eval_points))
    
def EmpCovariance(W, params):
    
    emp_cov = empirical_covariance(W, assume_centered=True)
    estimator = Covariance(cov=emp_cov)
    
    return estimator, {'covariance' : emp_cov}
