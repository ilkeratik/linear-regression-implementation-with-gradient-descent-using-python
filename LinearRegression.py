from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import math
from main import Main
class LinearRegression():

    def __init__(self, weights=None, b=0):
        self.W = weights
        self.b = b

    def rmse(self, predictions, actual):
        return np.sqrt(((predictions - actual) ** 2).mean())

    def MSEStep(self,X, y, learning_rate=0.01):
        """
        This function implements the gradient descent step for squared error as a
        performance metric.

        Parameters
        X : array of predictor features
        y : array of outcome values
        W : predictor feature coefficients
        b : regression function intercept
        learn_rate : learning rate

        Returns
        W_new : predictor feature coefficients following gradient descent step
        b_new : intercept following gradient descent step
        """
        #print(X.shape, self.W.shape,y.shape)
        y_pred = np.matmul(X,self.W)+self.b
        error = y-y_pred
        #print(error.shape)
        self.W = self.W + learning_rate* np.matmul(error,X)
        self.b = self.b + learning_rate* error.sum()
        
        curr_rmse = self.rmse(y,y_pred)
        return self.W, self.b, curr_rmse

    def visualize(self, X,y, regression_coefs,errors):
        X_min = X.min()
        X_max = X.max()
        fig, axs = plt.subplots(1, 2,figsize=(12, 8))
        counter = len(regression_coefs)
        # w,b = regression_coefs[0]
        # print(w,b)
        #print(regression_coefs)
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        axs[0].scatter(X, y)
        axs[0].ticklabel_format(useOffset=False)
        print(regression_coefs)
        for W, b in regression_coefs:
            counter -= 1
            color = [[0, 1 - 0.95 ** counter, 0]  if counter != 0 else [0.9, 0.0, 0.0] ][0]
            axs[0].plot(X, W*X+b, color = color)
        
        axs[1].set_xlabel('No of iteration')
        axs[1].set_ylabel('RMSE')
        axs[1].plot(errors)
        plt.show()
        
    def prediction(self,X):
        return self.stepFunction((np.matmul(X,self.W)+self.b)[0])

    def train_perceptron(self,X,y,learning_rate=0.01, num_ephocs=300):
        errors = []
        self.W = np.zeros(X.shape[1])
        regression_coefs = [np.hstack((self.W,self.b))]
        last_error = math.inf
        
        for i in range(num_ephocs):
            batch = np.random.choice(range(X.shape[0]), 15)
            X_batch = X[batch,:]
            y_batch = y[batch]
            W, b, err = self.MSEStep(X_batch,y_batch)
            errors.append(err)
            regression_coefs.append(np.hstack((W,[b])))
            if last_error == err:
                break
            last_error = err

        self.visualize(X,y,regression_coefs,errors)

