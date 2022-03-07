import numpy as np
import matplotlib.pyplot as plt

from main import Main
class Perceptron():

    def __init__(self, weights=[], b=np.random.rand(1)[0]):
        self.W = weights
        self.b = b

    def fit(self, X,y):
        x_min, x_max = np.min(X.T), np.max(X.T)
        self.W = np.array(np.random.rand(2,1))
        self.b = np.random.rand(1)[0] + x_max

        # x1 = X[:, 0]
        # x2 = X[:, 1]
        #color = ['red' if value == 1 else 'blue' for value in y]
        plt.scatter(x, y, marker='o', color='red')
        plt.xlabel('X input feature')
        plt.ylabel('y input feature')
        plt.title('Perceptron regression for X1, X2')
        plt.show()

    def stepFunction(t):
        if t >= 0:
            return 1
        return 0

    def prediction(self,X):
        return self.stepFunction((np.matmul(X,self.W)+self.b)[0])

    def train_perceptron(self,X,y,learning_rate=0.01, num_ephocs=25):

        for i in range(num_ephocs):
            W, b = self.perceptronStep(X,y,learning_rate=learning_rate)

    def perceptronStep(self,X,y,learning_rate=0.01):
        for i  in range(len(X)):

            if(self.prediction(X[i]) == 1 and y[i] == 0):
                self.W[0] = self.W[0] - learning_rate*X[i][0]
                self.W[1] = self.W[1] - learning_rate*X[i][1]
                self.b = self.b -learning_rate
            elif(self.prediction(X[i]) == 0 and y[i] == 1):
                self.W[0] = self.W[0] + learning_rate*X[i][0]
                self.W[1] = self.W[1] + learning_rate*X[i][1]
                self.b = self.b +learning_rate

        return self.W, self.b

ins = Perceptron()

x,y = Main().generateRandomData((100,1), range=(0,100))
print(y.T[1])
ins.fit(x,y)