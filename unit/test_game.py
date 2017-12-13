import unittest
import chess
from player.RandomPlayer import RandomPlayerMixin
from game import Game


class TestGame(unittest.TestCase):
    def test_repetition_number(self):
        game = Game(RandomPlayerMixin("random"), RandomPlayerMixin("random"))
        board = chess.Board()
        # white move knight
        m1 = chess.Move.from_uci("b1a3")
        board.push(m1)
        game.play(m1)
        self.assertEqual(game.count_repetitions(board), 1)
        # black move knight
        m2 = chess.Move.from_uci("b8c6")
        board.push(m2)
        game.play(m2)
        self.assertEqual(game.count_repetitions(board), 1)
        # white knight move back
        m3 = chess.Move.from_uci("a3b1")
        board.push(m3)
        game.play(m3)
        self.assertEqual(game.count_repetitions(board), 1)
        # black knight move back
        m4 = chess.Move.from_uci("c6b8")
        board.push(m4)
        game.play(m4)
        self.assertEqual(game.count_repetitions(board), 1)

        # Repeat mvoes
        # white move knight
        board.push(m1)
        game.play(m1)
        self.assertEqual(game.count_repetitions(board), 2)
        # black move knight
        board.push(m2)
        game.play(m2)
        self.assertEqual(game.count_repetitions(board), 2)
        # white knight move back
        board.push(m3)
        game.play(m3)
        self.assertEqual(game.count_repetitions(board), 2)
        # black knight move back
        board.push(m4)
        game.play(m4)
        self.assertEqual(game.count_repetitions(board), 2)

        # Repeat mvoes
        # white move knight
        board.push(m1)
        game.play(m1)
        self.assertEqual(game.count_repetitions(board), 3)
        # black move knight
        board.push(m2)
        move = game.play(m2)
        # Claim draw game
        self.assertIsNone(move)
        self.assertIsNone(game.winner_color())
