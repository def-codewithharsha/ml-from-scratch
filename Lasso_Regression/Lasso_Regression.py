# Loss Funtion : J(w,b)=1/m ​∑(y−y^​)**2+λ∑∣w∣
import numpy as np

class LassoRegression:
    def __init__(self,learning_rate=0.01,epochs=1000,lambda_=0.1):
        self.learning_rate = learning_rate
        self.epochs = 1000
        self.lambda_ = lambda_

    def fit(self,X,y):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.m, self.n = X.shape

        # Initialize weights and bias
        self.w = np.zeros(self.n)
        self.b = 0

        # Gradient Descent
        for _ in range(self.epochs):

            # Linear prediction
            y_pred = X.dot(self.w) + self.b

            # Errors
            error = y_pred - y

            # Gradient for weights (OLS + L1 penalty)
            dw = (2 / self.m) * X.T.dot(error) + self.lambda_ * np.sign(self.w)

            # Gradient for bias (bias is NOT regularized)
            db = (2 / self.m) * np.sum(error)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db