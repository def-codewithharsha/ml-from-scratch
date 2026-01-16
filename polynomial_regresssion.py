import numpy as np

class PolynomialRegression:
    def init(self,degree=2,learningrate=0.01,epochs=1000):
        
        self.degree = degree
        self.learningrate = learningrate
        self.epochs = epochs

    def _poly_features(self,X):
        return np.hstack([X**i for i in range(self.degree+1)])
    
    def fit(self,X,y):
        X_poly = self._poly_features(X)
        self.weights = np.zeros(X_poly.shape[1])

        for _ in range(self.epochs):
            y_pred = X_poly @ self.weights #[X_poly] DOT [self.weight]
            error= y_pred - y

            gradients = (2 / len(X))* (X_poly.T @ error)
            self.weights-=self.lr*gradients
    def predict(self, X):
        X_poly = self._poly_features(X)
        return X_poly @ self.weights
    