import numpy as np

class PolynomialRegression:
    def init(self,degree=2,learningrate=0.01,epochs=1000):
        
        self.degree = degree
        self.learningrate = learningrate
        self.epochs = epochs

    def _poly_features(self,X):
        return np.hstack([X**i for i in range(self.degree+1)])