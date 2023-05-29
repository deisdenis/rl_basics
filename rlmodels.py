import numpy as np


class EpsilonGreedy:
    _set_a = []
    _epsilon = 0.0
    _q = []
    _n = []
    _last_action = 0

    def __init__(self, set_a: list, epsilon: float = 0.0):
        self._set_a = set_a
        self._epsilon = epsilon
        self._q = np.zeros(len(set_a))
        self._n = np.zeros(len(set_a))

    def get_action(self):
        if np.random.random() < self._epsilon:
            a = np.random.choice(self._set_a)
        else:
            # TODO: argmax returns first max, we need arbitrary
            a = np.argmax(self._q)
        self._n[a] += 1
        self._last_action = a
        return a

    def process_result(self, reward):
        a = self._last_action  # readability only
        self._q[a] += (reward - self._q[a]) / self._n[a]
