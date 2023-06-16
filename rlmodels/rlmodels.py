from abc import ABC, abstractmethod
from spaces.space import Space

import numpy as np


class Agent(ABC):
    def __init__(self, agent_name, space: Space):
        # initial config
        self.agent_name = agent_name
        self._space = space
        self._actions = space.get_actions()
        self._states = space.get_states()
        self._terminal_states = space.get_terminal_states()

        # initialize Q
        self._q = np.zeros([len(self._states), len(self._actions)])

        # config specific variables
        self._alpha = 0.5
        self._epsilon = 0.1
        self._gamma = 1

        # episode storage configuration
        self._episodes = []
        self._episode_id = -1

    @abstractmethod
    def run_episode(self):
        pass

    def execute_action_id(self, action_id):
        next_state_id, reward, is_terminal = self._space.execute_action_id(action_id)
        self._episodes[self._episode_id].append((next_state_id, reward, is_terminal))
        return next_state_id, reward, is_terminal

    def make_epsilon_greedy_action(self, state_id, ignore_epsilon=False):
        # epsilon-greedy action selection
        if ignore_epsilon or np.random.random() > self._epsilon:
            return np.argmax(self._q[state_id, :])
        else:
            return np.random.choice(len(self._actions))

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_gamma(self, gamma):
        self._gamma = gamma


class WindyGridWorldAgent(Agent):

    def __init__(self, space: Space):
        super().__init__('SARSA on-policy TD control', space)
        self.set_alpha(0.5)
        self.set_epsilon(0.1)
        self.set_gamma(1)

    def run_episode(self):
        self._episode_id += 1
        self._episodes.append([])
        current_state = self._space.get_current_state()
        current_state_id = self._states.index(current_state)
        action = self.make_epsilon_greedy_action(current_state_id)

        safe_counter = 0
        is_terminal = False
        while not is_terminal and safe_counter < 100000:
            safe_counter += 1
            next_state_id, reward, is_terminal = self.execute_action_id(action)
            next_action = self.make_epsilon_greedy_action(next_state_id)
            self._q[current_state_id, action] += self._alpha * (
                        reward + self._gamma * self._q[next_state_id, next_action] - self._q[current_state_id, action])
            current_state_id = next_state_id
            action = next_action

    def get_policy(self, start):
        policy = []
        state_id = self._space.get_state_id_from_state(start)
        self._space.set_current_state(start)
        is_terminal = False
        while not is_terminal:
            action_id = np.argmax(self._q[state_id, :])
            policy.append(self._actions[action_id])
            state_id, _, is_terminal = self._space.execute_action_id(action_id)
        return policy


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
        self._states = None
        self._policy = np.random.choice(4, [5, 5])
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
        self._last_position = space.get_state().copy()
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
        new_pos = space.get_state().copy()
        new_pos_index = new_pos[0], new_pos[1]
        self._v_history[pos_index] = np.roll(self._v_history[pos_index], 1)
        self._v_history[pos_index][0] = reward
        self._v[pos_index] = np.dot(self._v_history[pos_index], self._discount_array) + self._v[
            new_pos_index] * self._discount
        # print(f'after index={index}')
        # print(f'self._n[index]={self._n[index]}')
        self._q[index] = (self._v[pos_index] - self._q[index]) / self._n[index]

    def action_id_to_readable(self):
        return self._actions[self._last_action_id]

    def action_readable_to_action_id(self, action_readable):
        return self._actions.index(action_readable)

    def policy_iteration(self, model):
        # initialization
        theta = 0.001  # improvement threshold
        self._states = [(i, j) for i in range(5) for j in range(5)]

        delta = 0
        policy_stable = False

        safety_counter_1 = 0
        while not policy_stable and safety_counter_1 < 100000:
            safety_counter_1 += 1

            # policy evaluation
            safety_counter_2 = 0
            while delta < theta and safety_counter_2 < 100:
                safety_counter_2 += 1
                delta = 0
                for state in self._states:
                    value = self._v[state]
                    reward, new_state = self.run_policy_from_state(model, state)
                    self._v[state] = reward + self._discount * self._v[new_state]
                    delta = max(delta, np.abs(value - self._v[state]))

            # policy improvement
            policy_stable = True
            for state in self._states:
                old_action = self._policy[state]
                policy_values = {}
                for action in self._actions:
                    reward, new_state = self.run_action_from_state(model, state, action)
                    policy_values[self.action_readable_to_action_id(action)] = \
                        reward + self._discount * self._v[new_state]
                self._policy[state] = max(policy_values, key=policy_values.get)
                if old_action != self._policy[state]:
                    policy_stable = False

        print(f'Evaluation completed in {safety_counter_1} steps')

    def run_policy_from_state(self, model, state):
        model.set_state(state)
        reward = model.process_action(self._policy[state])
        new_state = model.get_state_tuple()
        return reward, new_state

    def run_action_from_state(self, model, state, action):
        model.set_state(state)
        reward = model.process_action(action)
        new_state = model.get_state_tuple()
        return reward, new_state


class BlackJackAgent:
    _model = None

    def __init__(self, model):
        self._model = model
        self._v = {}.fromkeys(model.get_states(), 0)
        self._returns = {}
        for el in model.get_states():
            self._returns[el] = []

    def policy(self, state):
        if state < 20:
            return 1
        return 0

    def run_episode(self):
        model = self._model
        gamma = 1
        s = []
        a = []
        r = [0]
        cards = [model.hit(), model.hit()]
        t = 0
        playing = True
        while playing:
            s.append(min(model.get_score(cards), 22))
            a.append(self.policy(s[t]))
            if a[t] == 1:
                cards.append(model.hit())
                r.append(0)
            else:
                r.append(model.stick(cards))
                playing = False
            t += 1
        G = 0
        visited = []
        for step in range(t - 1, -1, -1):
            G = gamma * G + r[step + 1]
            if s[step] not in visited:
                returns = self._returns[s[step]]
                returns.append(G)
                self._v[s[step]] = sum(self._returns[s[step]]) / len(self._returns[s[step]])
                visited.append(s[step])

    def print_v(self):
        print(self._v)

# class WindyGridWorldAgent:
#     _states = []
#     _actions = []
#     _q = []
#
#     def __init__(self, model):
#         self._states = model.get_states()
#         self._actions = model.get_action_ids()
#         self._alpha = 0.5
#         self._epsilon = 0.1
#         self._gamma = 1
#         self._q = np.zeros([len(self._states), len(self._actions)])
#
#     def run_episode(self, terminal_states, model):
#         current_state = model.get_current_state()
#         current_state_id = self._states.index(current_state)
#         action = self._find_next_action(current_state_id)
#
#         safe_counter = 0
#         while current_state not in terminal_states and safe_counter < 100000:
#             safe_counter += 1
#             next_state, reward = model.use_action_id(action)
#             next_state_id = self._states.index(next_state)
#             next_action = self._find_next_action(next_state_id)
#             self._q[current_state_id, action] += self._alpha * (
#                     reward + self._gamma * self._q[next_state_id, next_action] - self._q[current_state_id, action])
#             current_state = next_state
#             current_state_id = next_state_id
#             action = next_action
#
#     def _find_next_action(self, state_id):
#         # epsilon-greedy action selection
#         if np.random.random() < self._epsilon:
#             action = np.random.choice(self._actions)
#         else:
#             action = np.argmax(self._q[state_id, :])
#         return action
#
#     def _find_best_action(self, state_id):
#         return np.argmax(self._q[state_id, :])
#
#     def get_policy(self, start, terminal_states, model):
#         policy = []
#         state = start
#         model.set_current_state(state)
#         while state not in terminal_states:
#             state_id = self._states.index(state)
#             action = np.argmax(self._q[state_id, :])
#             policy.append(action)
#             state, _ = model.use_action_id(action)
#         return policy
