import numpy as np


class EpsilonGreedy:
    _set_a = []
    _epsilon = 0.0
    _step_size = 1.0
    _q = []
    _q_init = []
    _n = []
    _rewards = []
    _last_action = 0
    _q_eval = 'simple'
    _storage_step = 1000
    _storage_erwa = 50
    _erwa = []

    def __init__(self, set_a: list, epsilon: float = 0.0, step_size: float = 1.0, q_eval: object = 'simple'):
        self._set_a = set_a
        self._epsilon = epsilon
        self._step_size = step_size
        self._q_eval = q_eval
        self._q = np.zeros(len(set_a))
        self._q_init = np.zeros(len(set_a))
        self._n = np.zeros(len(set_a), dtype=int)
        self._rewards = [np.zeros(self._storage_step) for _ in range(len(set_a))]
        self._erwa = [np.array([step_size * (1 - step_size) ** (n - i)
                               for i in range(n + 1)]) for n in range(self._storage_erwa)]

    def get_action(self):
        if np.random.random() < self._epsilon:
            a = np.random.choice(self._set_a)
        else:
            # TODO: argmax returns first max, we need arbitrary
            a = np.argmax(self._q)
        self._last_action = a
        return a

    def process_result(self, reward):
        a = self._last_action  # readability only
        self._rewards[a][self._n[a]] = reward
        self._check_storage(a, self._n[a])
        self._n[a] += 1
        if self._q_eval == 'simple':
            self._q[a] += (reward - self._q[a]) / self._n[a]
        if self._q_eval == 'exp_weights':
            if self._n[a] < self._storage_erwa:
                self._q[a] = (1 - self._step_size) ** self._n[a] * self._q_init[a] + np.dot(self._erwa[self._n[a]-1],
                                                                                            self._rewards[a][
                                                                                            :self._n[a]])
            else:
                self._q[a] = np.dot(self._erwa[self._storage_erwa-1],
                                    self._rewards[a][self._n[a]-self._storage_erwa:self._n[a]])

    def set_q(self, values: np.ndarray):
        self._q = values
        self._q_init = values

    def _check_storage(self, a, n):
        if n == self._rewards[a].shape[0]:
            self._rewards[a].resize((self._rewards[a].shape[0] + 1000))
