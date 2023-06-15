from abc import ABC, abstractmethod

import numpy as np


class Space(ABC):
    def __init__(self, space_name, states, actions, initial_state=None, terminal_states=None):
        # initial config
        self.space_name = space_name

        # initialize states
        self._states = states
        self._state_ids = np.arange(len(states))

        # initialize actions
        self._actions = actions
        self._action_ids = np.arange(len(actions))

        # initialize current_state
        if initial_state is not None:
            self._current_state = initial_state

        else:
            self._current_state = states[0]
        self._current_state_id = self._states.index(initial_state)

        # initialize terminal states (optional)
        self._terminal_states = terminal_states

    def set_states(self, states):
        self._states = states
        self._state_ids = np.arange(len(states))

    def set_actions(self, actions):
        self._actions = actions
        self._action_ids = np.arange(len(actions))

    def get_states(self):
        return self._states.copy()

    def get_state_ids(self):
        return self._state_ids.copy()

    def get_state_id_from_state(self, state):
        return self._states.index(state)

    def get_state_name_from_state_id(self, state_id):
        return self._states[state_id]

    def get_actions(self):
        return self._actions.copy()

    def get_action_ids(self):
        return self._action_ids.copy()

    def get_action_id_from_action(self, state):
        return self._states.index(state)

    def get_action_name_from_action_id(self, action_id):
        return self._actions[action_id]

    def get_current_state(self):
        return self._current_state

    def set_current_state(self, state):
        self._current_state = state

    def get_terminal_states(self):
        return self._terminal_states.copy()

    def execute_action_id(self, action_id):
        return self.execute_action(self._actions[action_id])

    @abstractmethod
    # returns state, reward, bool whether new state is a terminal state
    def execute_action(self, action) -> (int, float, bool):
        pass
