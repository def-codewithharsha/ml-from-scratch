import numpy as np

class MyLinearRegression:
    def __init__(self, learning_rate=0.01, num_of_iterations=1000, verbose=False):
        """
        Custom Linear Regression using Gradient Descent.
        Args:
            learning_rate (float): Step size for gradient descent.
            num_of_iterations (int): Number of iterations for training.
            verbose (bool): If True, prints cost during training.
        """
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Fit the model to training data.
        Args:
            X (ndarray): Feature matrix of shape (m, n).
            Y (ndarray): Target vector of shape (m,).
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Gradient Descent
        for i in range(self.num_of_iterations):
            self.update_weights()
            if self.verbose and i % 100 == 0:
                cost = self.compute_cost(self.Y, self.predict(self.X))
                print(f"Iteration {i}, Cost: {cost:.4f}")

    def update_weights(self):
        """
        Perform one step of gradient descent.
        """
        Y_prediction = self.predict(self.X)

        # Gradients
        dw = (2 * self.X.T.dot(Y_prediction - self.Y)) / self.m
        db = (2 * np.sum(Y_prediction - self.Y)) / self.m

        # Update weights
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for given input.
        Args:
            X (ndarray): Feature matrix of shape (m, n).
        Returns:
            ndarray: Predicted values.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.dot(self.w) + self.b

    def compute_cost(self, Y, Y_prediction):
        """
        Compute Mean Squared Error cost.
        """
        return np.mean((Y - Y_prediction) ** 2)