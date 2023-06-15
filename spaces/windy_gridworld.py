from spaces.space import Space


class WindyGridWorld(Space):

    def __init__(self):
        super().__init__(space_name='WindyGridWorld',
                       states=[(i, j) for i in range(7) for j in range(10)],
                       actions=['up', 'down', 'left', 'right'],
                       initial_state=(3, 0),
                       terminal_states=[(3, 7)])

    def _add_wind(self, new_j):
        if new_j in [3, 4, 5, 8]:
            return -1
        elif new_j in [6, 7]:
            return -2
        else:
            return 0

    def execute_action(self, action):
        i, j = self._current_state
        i = max(0, i + self._add_wind(j))
        if action == 'up':
            i = max(0, i - 1)
        elif action == 'down':
            i = min(6, i + 1)
        elif action == 'left':
            j = max(0, j - 1)
        elif action == 'right':
            j = min(9, j + 1)
        else:
            Exception("Unknown action")
        self._current_state = (i, j)
        if self._current_state == (3, 7):
            return self.get_state_id_from_state(self._current_state), 1.0, True
        else:
            return self.get_state_id_from_state(self._current_state), -0.1, False

# class WindyGridWorld:
#     _states = []
#
#     def __init__(self):
#         self._states = [(i, j) for i in range(7) for j in range(10)]
#         self._actions = ['up', 'down', 'left', 'right']
#         self._current_state = (3, 0)
#         self._terminal_states = [(3, 7)]
#
#     def get_states(self):
#         return self._states.copy()
#
#     def get_current_state(self):
#         return self._current_state
#
#     def get_terminal_states(self):
#         return self._terminal_states
#
#     def set_current_state(self, state):
#         self._current_state = state
#
#     def get_actions(self):
#         return self._actions.copy()
#
#     def get_action_ids(self):
#         return [self.action_to_action_id(action) for action in self._actions]
#
#     def action_to_action_id(self, action):
#         return self._actions.index(action)
#
#     def use_action_id(self, action_id):
#         return self.use_action_name(self._actions[action_id])
#
#     def _add_wind(self, new_j):
#         if new_j in [3, 4, 5, 8]:
#             return -1
#         elif new_j in [6, 7]:
#             return -2
#         else:
#             return 0
#
#     def use_action_name(self, action) -> (object, float):
#         i, j = self._current_state
#         i = max(0, i + self._add_wind(j))
#         if action == 'up':
#             i = max(0, i - 1)
#         elif action == 'down':
#             i = min(6, i + 1)
#         elif action == 'left':
#             j = max(0, j - 1)
#         elif action == 'right':
#             j = min(9, j + 1)
#         else:
#             Exception("Unknown action")
#         self._current_state = (i, j)
#         if self._current_state == (3, 7):
#             return self._current_state, 1
#         else:
#             return self._current_state, -0.1
