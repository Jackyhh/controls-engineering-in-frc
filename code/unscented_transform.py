import numpy as np


def unscented_transform(States, CovDim, sigmas, Wm, Wc, noiseCov):
    x = Wm @ sigmas
    y = sigmas - x[np.newaxis, :]
    P = y.T @ np.diag(Wc) @ y
    P += noiseCov

    return x, P
