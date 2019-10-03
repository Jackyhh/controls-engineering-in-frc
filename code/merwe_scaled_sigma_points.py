import numpy as np
import scipy as sp

class MerweScaledSigmaPoints:
    def __init__(states, alpha = 1e-3, beta = 2, kappa = 3 - states:
        self.states = states
        self.alpha = alpha
        self.kappa = kappa

        self.Wm = np.zeros((1, 2 * self.states + 1))
        self.Wc = np.zeros((1, 2 * self.states + 1))

        self.__compute_weights(beta)

    def num_sigmas():
        return 2 * self.states + 1

    def sigma_points(x, P):
        l = self.alpha ** 2 * (self.states + self.kappa) - self.states
        U = sp.linalg.cholesky((l + self.states) * P)

        sigmas = np.zeros((2 * self.states + 1, self.states))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = x + U[k]
            sigmas[self.states + k + 1] = x - U[k]

        return sigmas

    def Wm():
        return self.Wm

    def Wm(i):
        return self.Wm[0, i]

    def Wc():
        return self.Wc

    def Wc(int i):
        return self.Wc[0, i]

    def __compute_weights(double beta):
        l = self.alpha ** 2 * (self.states + self.kappa) - self.states

        c = 0.5 / (self.states + l)
        self.Wc = np.full((1, 2 * self.states + 1), c)
        self.Wm = np.full((1, 2 * self.states + 1), c)
        for i in range(2 * self.states + 1):
            self.Wc(0, i) =
                l / (self.states + l) + (1 - self.alpha ** 2 + beta)
            self.Wm(0, i) = l / (self.states + l)
