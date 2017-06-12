#转载至：Python Machine Learning.
import numpy as np
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    -------------------
    eta:float
        learning rate(between 0.0 to 1.0)
    n_iter:int
        passes over the dataset
    
    attributes
    -------------------
    w_:1d-array
        weights after fitting
    errors:list
        number of misclassifications in each epoch
        
    """
    
    def __init__(self, eta = 0.01, n_iter = 10 ):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, x, y):
        """fit training data
            
        parameters
        --------------
        x: {array-like}, shape = {n_samples, n_features} 
            training vectors, where n_samples is the number of samples and 
            n_features is the number of feature.
        y: array-like, shape = {n_samples}
            target values.
            
        Returns
        -------------
        self:Object
            
        """
        
        self.w_ = np.zeros(1 + x.shape[1]) #Add w_0
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        """calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1) #analoge ? : in C++