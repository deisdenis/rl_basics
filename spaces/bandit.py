import numpy as np


class Bandit:
    _loc = 0.0
    _scale = 1.0
    _k = 0
    _means = []

    def __init__(self, loc: float = 0.0, scale: float = 1.0, k: object = 10) -> object:
        self._loc = loc
        self._scale = scale
        self._k = k
        self._means = np.random.normal(loc, scale, k)

    def use_lever(self, lever):
        return np.random.normal(self._means[lever], self._scale)

    def get_optimal_action(self):
        return np.argmax(self._means)
