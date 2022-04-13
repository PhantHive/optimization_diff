import copy

import numpy as np


class FixedGradient:
    '''
    Gradient descent with fixed step
    '''

    def __init__(self, A, b, x0, tol, step, iMax):
        '''
        :param A: matrix
        :param b: vector
        :param x0:
        :param tol: tolerance
        :param step: self explanatory
        :param iMax: max number of iteration
        '''

        self.A = A
        self.b = b
        self.x = x0
        self.tol = tol
        self.step = step
        self.iMax = iMax
        self.xit = [x0]


    def resolve(self):

        i = 0
        Rk = self.tol*2
        while np.linalg.norm(Rk) > self.tol and i < self.iMax:
            Rk = self.A@self.x - self.b
            Dk = - Rk
            self.x += self.step*Dk
            i += 1
            self.xit.append(self.x.copy())

        return self.x, self.xit, i



if __name__ == '__main__':

    dataP = np.loadtxt("../data/dataP.dat")
    dataQ = np.loadtxt("../data/dataQ.dat")

    X = []
    for i in range(len(dataP)):
        X.append([1, dataP[i]])

    # Parameters
    A = np.transpose(X) @ X
    b = np.transpose(X) @ dataQ
    x0 = np.transpose(np.array([-9., -7.]))
    tol = 10**-6
    step = 10**-3
    iMax = 10000

    fG = FixedGradient(A, b, x0, tol, step, iMax)
    x, xit, i = fG.resolve()

    print(x, xit, i)
