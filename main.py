import random as r
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
class Main:

    def __init__(self):
        pass
    
    def generateRandomData(self,size,range=None,dtype='float', dist='uniform',beta_0=2, beta_1=.4):
        """
        parameters:
        size: tuple(number of instances, dimension of an instance)
        range: interval for random numbers to create (min,max)
        dtype: data type of an instance could be [float,int]
        dist: uniform or normal distribution

        returns array of the data created
        """
        
        res = None
        if range:
            if dtype == 'float':
                res = np.random.randn(*size)
                (min,max) = range
                res = (max-min) * res +min

            elif dtype == 'int':
                (min,max) = range
                res = np.random.randint(min,max+1, size=size)

            else:
                raise TypeError('Unsupported data type')

        else:
            if dtype == 'float':
                res = np.random.randn(*size)
            elif dtype == 'int':
                res = np.random.randint(0,100, size=size)
            else:
                raise TypeError('Unsupported data type')

        #print(res[:,0])
        e = np.random.uniform(-5,5, size=size[0])
        # e = np.random.randn(size[0])
        # e = 20*e -10
        
        
        xs= beta_0 + beta_1* res[:,0]
        y = beta_0 + beta_1* res[:,0].reshape(res[:,0].shape[0],) + e
        return res, y

    def generateData(self,n,beta_0=2, beta_1=.4):
        x = np.linspace(1,n,2*n)
        e = np.random.uniform(-10,10, size=2*n)
        y = beta_0 + beta_1* x + e
        x = x.reshape(x.shape[0],1)
        return x,y

    def dataset_to_csv(self,x,y):
        np.savetxt('data_generated.csv', np.column_stack((x,y)), delimiter=',',fmt='%1.4f')

if __name__=='__main__':

    m = Main()
    x,y = m.generateRandomData((300,1), range=(0,100))
    #x,y = m.generateData(200)
    print(x.shape, y.shape)

    # data = np.loadtxt('data.csv', delimiter = ',',skiprows=1)
    # x = data[:,:-1]
    # y = data[:,-1]

    # normalizing the values; this is a must, otherwise gradient descent will explode and won't converge
    lr = LinearRegression()
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    lr.train_perceptron(x,y,num_ephocs=500,learning_rate=0.1)