import numpy as np


class OptimalGradient:

    def __init__(self, A, b, x0, iMax, tol):
        self.A = A
        self.b = b
        self.x = x0
        self.iMax = iMax
        self.tol = tol
        self.xit = [x0]

    def resolve(self):
        i = 0
        Rk = self.tol * 2
        while i < self.iMax and np.linalg.norm(Rk) > self.tol:
            i += 1
            Rk = self.A @ self.x - self.b
            step = np.linalg.norm(Rk) ** 2 / (np.transpose(Rk) @ self.A @ Rk)
            self.x += -step * Rk
            self.xit.append(self.x)

        return self.x, self.xit, i


if __name__ == '__main__':

    dataP = np.loadtxt("../data/dataP.dat")
    dataQ = np.loadtxt("../data/dataQ.dat")
    # print(dataP)
    # print(dataQ)

    X = list()
    for i in range(len(dataP)):
        X.append([1, dataP[i]])

    # print(X)
    A = np.transpose(X) @ X
    b = np.transpose(X) @ dataQ
    x0 = np.transpose(np.array([-9., -7.]))
    iMax = 10000
    tol = 10 ** -6

    oG = OptimalGradient(A, b, x0, iMax, tol)
    x, xit, i = oG.resolve()
    print(x, xit, i)
