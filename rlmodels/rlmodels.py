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
                self._q[a] = (1 - self._step_size) ** self._n[a] * self._q_init[a] + np.dot(self._erwa[self._n[a] - 1],
                                                                                            self._rewards[a][
                                                                                            :self._n[a]])
            else:
                self._q[a] = np.dot(self._erwa[self._storage_erwa - 1],
                                    self._rewards[a][self._n[a] - self._storage_erwa:self._n[a]])

    def set_q(self, values: np.ndarray):
        self._q = values
        self._q_init = values

    def _check_storage(self, a, n):
        if n == self._rewards[a].shape[0]:
            self._rewards[a].resize((self._rewards[a].shape[0] + 1000))


class GridWorldAgent:
    _last_position = [0, 0]
    _last_action_id = 0
    _epsilon = 1
    _trial = 0

    def __init__(self):
        self._actions = ['up', 'down', 'left', 'right']
        self._q = np.zeros([5, 5, 4])  # x, y - state, z - action
        self._n = np.zeros([5, 5, 4])  # x, y - state, z - action
        self._r = np.zeros([5, 5, 4, 50])  # x, y - state, z - action, i - last 50 rewards
        self._v = np.zeros([5, 5])  # x, y - state
        self._v_history = np.zeros([5, 5, 50])  # x, y - state
        self._epsilon = 1
        self._discount = 0.9
        self._discount_array = np.array([self._discount ** n for n in range(1, 51)])

    def get_action(self, space):
        self._last_position = space.get_position().copy()
        pos_index = self._last_position[0], self._last_position[1]
        if np.random.random() < 1 - np.tanh(self._trial / 100000.0):
            self._last_action_id = np.random.choice([0, 1, 2, 3])
        else:
            self._last_action_id = np.unravel_index(np.argmax(self._q[pos_index], axis=None), self._q.shape)[2]
        index = self._last_position[0], self._last_position[1], self._last_action_id
        self._n[index] += 1
        # print(f'before index={index}')
        self._trial += 1
        return self.action_id_to_readable()

    def process_result(self, reward, space):
        index = self._last_position[0], self._last_position[1], self._last_action_id
        pos_index = self._last_position[0], self._last_position[1]
        new_pos = space.get_position().copy()
        new_pos_index = new_pos[0], new_pos[1]
        self._v_history[pos_index] = np.roll(self._v_history[pos_index], 1)
        self._v_history[pos_index][0] = reward
        self._v[pos_index] = np.dot(self._v_history[pos_index], self._discount_array) + self._v[new_pos_index] * self._discount
        # print(f'after index={index}')
        # print(f'self._n[index]={self._n[index]}')
        self._q[index] = (self._v[pos_index] - self._q[index]) / self._n[index]

    def action_id_to_readable(self):
        if self._last_action_id == 0:
            return 'up'
        if self._last_action_id == 1:
            return 'down'
        if self._last_action_id == 2:
            return 'left'
        if self._last_action_id == 3:
            return 'right'
