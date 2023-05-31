import numpy as np


class BlackJack:

    def __init__(self):
        self._deck = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4)

    def hit(self):
        return np.random.choice(self._deck)

    def get_score(self, player_cards):
        player_score = sum(player_cards)
        while player_score > 21 and 11 in player_cards:
            player_cards.remove(11)
            player_cards.append(1)
            player_score = sum(player_cards)
        return player_score

    def stick(self, player_cards):
        dealer_cards = [self.hit(), self.hit()]
        while self.get_score(dealer_cards) < 17:
            dealer_cards.append(self.hit())
        dealer_score = self.get_score(dealer_cards)
        player_score = self.get_score(player_cards)
        if player_score > 21:
            return -1
        elif player_score <= 21 < dealer_score:
            return 1
        elif player_score == dealer_score:
            return 0
        elif player_score > dealer_score:
            return 1
        elif dealer_score > player_score:
            return -1
        else:
            Exception("Undefined condition")

    def get_states(self):
        return np.arange(4, 23)


