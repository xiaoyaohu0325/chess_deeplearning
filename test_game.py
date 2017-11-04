from game import Game
from player import AIPlayer
from policy import ResnetPolicy
import unittest
import chess


class TestGame(unittest.TestCase):
    def test_play(self):
        policy_1 = ResnetPolicy.load_model("./out/model.json")
        policy_1.model.load_weights("./out/random_weights")
        policy_2 = ResnetPolicy.load_model("./out/model.json")
        policy_2.model.load_weights("./out/train/weights.00000.hdf5")

        p1 = AIPlayer('gen0', chess.WHITE, policy_1, simulation=40, depth=2)
        p2 = AIPlayer('gen1', chess.BLACK, policy_2, simulation=40, depth=2)
        game = Game(p1, p2)

        print('game start')
        move = game.play()
        while move is not None:
            print(move.uci())
            move = game.play()

        print('game over')
        print('winner: ', game.winner_color())
