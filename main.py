import random as r
import numpy as np
import matplotlib as plt
class Main:

    def __init__(self):
        pass
    
    def generateRandomData(self,size,range=None,dtype='float', dist='uniform'):
        """
        parameters:
        size: tuple(number of instances, dimension of an instance)
        range: interval for random numbers to create (min,max)
        dtype: data type of an instance could be [float,int]
        dist: uniform or normal

        returns array of the data created
        """
        
        res = None

        if range:
            if dtype == 'float':
                res = np.random.rand(zip(*size))
                (min,max) = range
                res = (max-min) * res +min

            elif dtype == 'int':
                (min,max) = range
                res = np.random.randint(min,max+1, size=size)

            else:
                raise('Unsupported data type')

        else:
            if dtype == 'float':
                res = np.random.rand(zip(*size))
            elif dtype == 'int':
                res = np.random.randint(0,100, size=size)
            else:
                raise('Unsupported data type')

        return res
if __name__=='__main__':

    m = Main()
    data = m.generateRandomData((3,3))
    print(data)
    
    data = m.generateRandomData((5,3), range=(10,15), dtype='int')
    print(data)