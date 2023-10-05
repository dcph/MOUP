import numpy as np


class GaussianNoise:


    def __init__(self, mu=0.0, sigma=1.0):
        self._mu = mu
        assert 0 <= sigma, "Noise std should not be negative."
        self._sigma = sigma

    def __call__(self, size):
        return np.random.normal(self._mu, self._sigma, size)


class OUNoise:


    def __init__(self, mu=0.0, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * np.sqrt(dt)
        self._x0 = x0
        self.reset()

    def reset(self) -> None:

        self._x = self._x0

    def __call__(self, size, mu=None):

        if self._x is None or isinstance(
            self._x, np.ndarray
        ) and self._x.shape != size:
            self._x = 0.0
        if mu is None:
            mu = self._mu
        r = self._beta * np.random.normal(size=size)
        self._x = self._x + self._alpha * (mu - self._x) + r
        return self._x  # type: ignore