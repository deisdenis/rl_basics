class GridWorld:

    def __init__(self):
        self._position = [0, 0]
        self._fancy_positions = [[0, 1], [0, 3]]

    def get_position(self):
        return self._position

    def process_action(self, action: object) -> float:
        if not self.check_action(action):
            return -1
        elif self._position == [0, 1]:
            self._position = [4, 1]
            return 10
        elif self._position == [0, 3]:
            self._position = [2, 3]
            return 5
        elif action == 'up':
            self._position[0] -= 1
        elif action == 'down':
            self._position[0] += 1
        elif action == 'left':
            self._position[1] -= 1
        elif action == 'right':
            self._position[1] += 1
        return -0.1

    def check_action(self, action: object) -> bool:
        if self._position in self._fancy_positions:
            return True
        if (action == 'up' and self._position[0] == 0) \
                or (action == 'down' and self._position[0] == 4) \
                or (action == 'left' and self._position[1] == 0) \
                or (action == 'right' and self._position[1] == 4):
            return False
        else:
            return True
