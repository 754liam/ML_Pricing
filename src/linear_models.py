import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

class OLS:
    def __init__(self):
        self.weights = None


    def fit(self, X, Y):
        # find weights in the form of a 1D numpy array such that, for each stock, multiply them by feature_count weights 
        # where each column has its own weight. Thus, there will be a feature_count amount of weights - and a row_number 
        # amount of multiplications for each weight. The sum of all of these weights and values in the columns
        # is being best adjusted to equal the return. These weights are the closest possible fit - a minimization of the sum of
        # squared errors between X @ w and Y. This is OLS.
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ Y
    

    
    