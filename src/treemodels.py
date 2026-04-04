import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

class RandomForestModel:
    # fitted flag
    fitted = False
    
    # We begin by declaring a Random Forest model via sklearn with the default
    # number of trees (or estimators) as 100, and the default depth of the decision trees
    # being 5.
    def __init__(self, n_estimators=100, max_depth=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    
    # Basic fitting (well, not basic, complicated behind the scenes) instantiating our models attributes
    def fit(self, X, Y):
        self.model.fit(X, Y)
        self.fitted = True
    # decisions likely look like: is mom1m > 0.2, is vol3 > 0.5, etc - keep in mind that features are normalized -1 to 1

    # Prediction given arbritrary input X where X is expected to be of similar form to that fitted
    def predict(self, X):
        if self.fitted == False:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        predictions = self.model.predict(X)
        return predictions
    # in sum, each stock enters n_estimators trees, walks through a learned
    # decision tree, and the decision trees ret prediction is summed up with all 
    # other decision tree ret predictions, then averaged, and then that is the final
    # ret prediction for that stock. 

class GradientBoostedModel:
    # fitted flag
    fitted = False

    # basically keeps making trees (just as in random forests) which are informed from former trees,
    # the final prediction is the sum of all trees - and learning_rate is how much each tree contributes.

    # max_iter is how many new rounds of correction; max depth is how many decisions are made per tree;
    # learning_rate is useful for controlling overshooting. each tree corrects the former trees error my
    # a continuing fraction of the remaining gap during training/fitting

    # in a way, its sort of like a recursive version of OLS, where each new tree is trained on the
    # error of its former
    def __init__(self, max_iter=100, max_depth=3, learning_rate=0.01):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    def fit(self, X, Y):
        self.model.fit(X,Y)
        self.fitted = True
    def predict(self, X):
        if self.fitted == False:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        predictions = self.model.predict(X)
        return predictions
        
    