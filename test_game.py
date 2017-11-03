from game import Game
from player import RandomPlayer, AIPlayer
from policy import ResnetPolicy
import unittest
import chess


class TestGame(unittest.TestCase):
    def test_play(self):
        policy = ResnetPolicy.load_model("./out/model.json")
        p1 = RandomPlayer('random', chess.WHITE)
        p2 = AIPlayer('ai', chess.BLACK, policy, simulation=1, depth=1)
        game = Game(p1, p2)

        print('game start')
        move = game.play()
        while move is not None:
            print(move.uci())
            move = game.play()

        print('game over')
        print('winner: ', game.winner_color())
