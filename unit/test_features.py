import unittest
import chess
import numpy as np
from util.features import extrac_piece_planes, extract_features
from player.RandomPlayer import RandomPlayerMixin
from player.Node import Node
from game import Game


class TestFeatures(unittest.TestCase):
    def test_extract_piece_planes_white(self):
        board = chess.Board(fen="1n2kb2/4p3/b1p3rp/5PP1/pP1p2P1/N2n3P/PB1P2K1/1r2QBNR w - - 22 51")
        """Orientation is white
        . n . . k b . .
        . . . . p . . .
        b . p . . . r p
        . . . . . P P .
        p P . p . . P .
        N . . n . . . P
        P B . P . . K .
        . r . . Q B N R
        """
        game = Game(RandomPlayerMixin("random"), RandomPlayerMixin("random"))
        game.turn = chess.WHITE
        planes = extrac_piece_planes(game, board)
        self.assertEqual(planes.shape, (8, 8, 14))
        """
        White pieces
        0. pawn
        1. knight
        2. bishop
        3. rook
        4: queue
        5: king
        """
        # seven white pawns, a2, d2, h3, b4, g4, f5, g5
        self.assertEqual(np.sum(planes[:, :, 0]), 7)
        self.assertEqual(planes[1, 0, 0], 1)  # a2
        self.assertEqual(planes[1, 3, 0], 1)  # d2
        self.assertEqual(planes[2, 7, 0], 1)  # h3
        self.assertEqual(planes[3, 1, 0], 1)  # b4
        self.assertEqual(planes[3, 6, 0], 1)  # g4
        self.assertEqual(planes[4, 5, 0], 1)  # f5
        self.assertEqual(planes[4, 6, 0], 1)  # g5
        # two white knights, g1, a3
        self.assertEqual(np.sum(planes[:, :, 1]), 2)
        self.assertEqual(planes[0, 6, 1], 1)  # g1
        self.assertEqual(planes[2, 0, 1], 1)  # a3
        # two white bishops, f1, b2
        self.assertEqual(np.sum(planes[:, :, 2]), 2)
        self.assertEqual(planes[0, 5, 2], 1)  # f1
        self.assertEqual(planes[1, 1, 2], 1)  # b2
        # one white rook, h1
        self.assertEqual(np.sum(planes[:, :, 3]), 1)
        self.assertEqual(planes[0, 7, 3], 1)  # h1
        # one white queue, e1
        self.assertEqual(np.sum(planes[:, :, 4]), 1)
        self.assertEqual(planes[0, 4, 4], 1)  # e1
        # one white king, g2
        self.assertEqual(np.sum(planes[:, :, 5]), 1)
        self.assertEqual(planes[1, 6, 5], 1)  # g2
        """
        Black pieces
        6. pawn
        7. knight
        8. bishop
        9. rook
        10: queue
        11: king
        """
        # five black pawns, e7, c6, h6, a4, d4
        self.assertEqual(np.sum(planes[:, :, 6]), 5)
        self.assertEqual(planes[6, 4, 6], 1)  # e7
        self.assertEqual(planes[5, 2, 6], 1)  # c6
        self.assertEqual(planes[5, 7, 6], 1)  # h6
        self.assertEqual(planes[3, 0, 6], 1)  # a4
        self.assertEqual(planes[3, 3, 6], 1)  # d4
        # two black knights, b8, d3
        self.assertEqual(np.sum(planes[:, :, 7]), 2)
        self.assertEqual(planes[7, 1, 7], 1)  # b8
        self.assertEqual(planes[2, 3, 7], 1)  # d3
        # two black bishops, f8, a6
        self.assertEqual(np.sum(planes[:, :, 8]), 2)
        self.assertEqual(planes[7, 5, 8], 1)  # f8
        self.assertEqual(planes[5, 0, 8], 1)  # a6
        # two black rooks, g6, b1
        self.assertEqual(np.sum(planes[:, :, 9]), 2)
        self.assertEqual(planes[5, 6, 9], 1)  # g6
        self.assertEqual(planes[0, 1, 9], 1)  # b1
        # zero black queue
        self.assertEqual(np.sum(planes[:, :, 10]), 0)
        # one black king, e8
        self.assertEqual(np.sum(planes[:, :, 11]), 1)
        self.assertEqual(planes[7, 4, 11], 1)  # e8

    def test_extract_piece_planes_black(self):
        board = chess.Board(fen="1n2kb2/4p3/b1p3rp/5PP1/pP1p2P1/N2n3P/PB1P2K1/1r2QBNR b - - 22 51")
        """Orientation is black
        R N B Q . . r .
        . K . . P . B P
        P . . . n . . N
        . P . . p . P p
        . P P . . . . .
        p r . . . p . b
        . . . p . . . .
        . . b k . . n .
        """
        game = Game(RandomPlayerMixin("random"), RandomPlayerMixin("random"))
        game.turn = chess.BLACK
        planes = extrac_piece_planes(game, board)
        self.assertEqual(planes.shape, (8, 8, 14))
        """
        Black pieces
        0. pawn
        1. knight
        2. bishop
        3. rook
        4: queue
        5: king
        """
        # five black pawns, e7, c6, h6, a4, d4
        self.assertEqual(np.sum(planes[:, :, 0]), 5)
        self.assertEqual(planes[1, 3, 0], 1)  # e7
        self.assertEqual(planes[2, 5, 0], 1)  # c6
        self.assertEqual(planes[2, 0, 0], 1)  # h6
        self.assertEqual(planes[4, 7, 0], 1)  # a4
        self.assertEqual(planes[4, 4, 0], 1)  # d4
        # two black knights, b8, d3
        self.assertEqual(np.sum(planes[:, :, 1]), 2)
        self.assertEqual(planes[0, 6, 1], 1)  # b8
        self.assertEqual(planes[5, 4, 1], 1)  # d3
        # two black bishops, f8, a6
        self.assertEqual(np.sum(planes[:, :, 2]), 2)
        self.assertEqual(planes[0, 2, 2], 1)  # f8
        self.assertEqual(planes[2, 7, 2], 1)  # a6
        # two black rooks, g6, b1
        self.assertEqual(np.sum(planes[:, :, 3]), 2)
        self.assertEqual(planes[2, 1, 3], 1)  # g6
        self.assertEqual(planes[7, 6, 3], 1)  # b1
        # zero black queue
        self.assertEqual(np.sum(planes[:, :, 4]), 0)
        # one black king, e8
        self.assertEqual(np.sum(planes[:, :, 5]), 1)
        self.assertEqual(planes[0, 3, 5], 1)  # e8
        """
        White pieces
        6. pawn
        7. knight
        8. bishop
        9. rook
        10: queue
        11: king
        """
        # seven white pawns, a2, d2, h3, b4, g4, f5, g5
        self.assertEqual(np.sum(planes[:, :, 6]), 7)
        self.assertEqual(planes[6, 7, 6], 1)  # a2
        self.assertEqual(planes[6, 4, 6], 1)  # d2
        self.assertEqual(planes[5, 0, 6], 1)  # h3
        self.assertEqual(planes[4, 6, 6], 1)  # b4
        self.assertEqual(planes[4, 1, 6], 1)  # g4
        self.assertEqual(planes[3, 2, 6], 1)  # f5
        self.assertEqual(planes[3, 1, 6], 1)  # g5
        # two white knights, g1, a3
        self.assertEqual(np.sum(planes[:, :, 7]), 2)
        self.assertEqual(planes[7, 1, 7], 1)  # g1
        self.assertEqual(planes[5, 7, 7], 1)  # a3
        # two white bishops, f1, b2
        self.assertEqual(np.sum(planes[:, :, 8]), 2)
        self.assertEqual(planes[7, 2, 8], 1)  # f1
        self.assertEqual(planes[6, 6, 8], 1)  # b2
        # one white rook, h1
        self.assertEqual(np.sum(planes[:, :, 9]), 1)
        self.assertEqual(planes[7, 0, 9], 1)  # h1
        # one white queue, e1
        self.assertEqual(np.sum(planes[:, :, 10]), 1)
        self.assertEqual(planes[7, 3, 10], 1)  # e1
        # one white king, g2
        self.assertEqual(np.sum(planes[:, :, 11]), 1)
        self.assertEqual(planes[6, 1, 11], 1)  # g2

    def test_extract_feature(self):
        game = Game(RandomPlayerMixin("random"), RandomPlayerMixin("random"))
        node = Node(board=chess.Board())
        # white move knight
        m1 = chess.Move.from_uci("b1a3")
        node = node.append_child_node(m1, 0.5)
        node.play_move()
        game.play(m1)

        self.assertEqual(game.turn, chess.BLACK)
        features = extract_features(game, node)
        self.assertEqual(features.shape, (8, 8, 119))
        # previous six steps are empty
        self.assertEqual(np.sum(features[:, :, 0:84]), 0)
        # 84 to 98 is the start position of a game, orientation is black
        planes = features[:, :, 84:98]
        """
        White pieces
        6. pawn
        7. knight
        8. bishop
        9. rook
        10: queue
        11: king
        """
        # seven white pawns, a2-h2
        self.assertEqual(np.sum(planes[:, :, 6]), 8)
        self.assertEqual(planes[6, 0, 6], 1)
        self.assertEqual(planes[6, 1, 6], 1)
        self.assertEqual(planes[6, 2, 6], 1)
        self.assertEqual(planes[6, 3, 6], 1)
        self.assertEqual(planes[6, 4, 6], 1)
        self.assertEqual(planes[6, 5, 6], 1)
        self.assertEqual(planes[6, 6, 6], 1)
        self.assertEqual(planes[6, 7, 6], 1)
        # two white knights, g2, b2
        self.assertEqual(np.sum(planes[:, :, 7]), 2)
        self.assertEqual(planes[7, 1, 7], 1)
        self.assertEqual(planes[7, 6, 7], 1)
        # two white bishops, f1, c1
        self.assertEqual(np.sum(planes[:, :, 8]), 2)
        self.assertEqual(planes[7, 2, 8], 1)
        self.assertEqual(planes[7, 5, 8], 1)
        # one white rook, h1, a1
        self.assertEqual(np.sum(planes[:, :, 9]), 2)
        self.assertEqual(planes[7, 0, 9], 1)
        self.assertEqual(planes[7, 7, 9], 1)
        # one white queue, d1
        self.assertEqual(np.sum(planes[:, :, 10]), 1)
        self.assertEqual(planes[7, 4, 10], 1)
        # one white king, e1
        self.assertEqual(np.sum(planes[:, :, 11]), 1)
        self.assertEqual(planes[7, 3, 11], 1)

        # 98 to 112 is the state after first move, orientation is black
        planes = features[:, :, 98:112]
        """
        Black pieces
        0. pawn
        1. knight
        2. bishop
        3. rook
        4: queue
        5: king
        """
        # five black pawns, a7-h7
        self.assertEqual(np.sum(planes[:, :, 0]), 8)
        self.assertEqual(planes[1, 0, 0], 1)
        self.assertEqual(planes[1, 1, 0], 1)
        self.assertEqual(planes[1, 2, 0], 1)
        self.assertEqual(planes[1, 3, 0], 1)
        self.assertEqual(planes[1, 4, 0], 1)
        self.assertEqual(planes[1, 5, 0], 1)
        self.assertEqual(planes[1, 6, 0], 1)
        self.assertEqual(planes[1, 7, 0], 1)
        # two black knights, b8, g8
        self.assertEqual(np.sum(planes[:, :, 1]), 2)
        self.assertEqual(planes[0, 1, 1], 1)
        self.assertEqual(planes[0, 6, 1], 1)
        # two black bishops, c8, f8
        self.assertEqual(np.sum(planes[:, :, 2]), 2)
        self.assertEqual(planes[0, 2, 2], 1)
        self.assertEqual(planes[0, 5, 2], 1)
        # two black rooks, a8, h8
        self.assertEqual(np.sum(planes[:, :, 3]), 2)
        self.assertEqual(planes[0, 0, 3], 1)
        self.assertEqual(planes[0, 7, 3], 1)
        # zero black queue, d8
        self.assertEqual(np.sum(planes[:, :, 4]), 1)
        self.assertEqual(planes[0, 4, 4], 1)
        # one black king, e8
        self.assertEqual(np.sum(planes[:, :, 5]), 1)
        self.assertEqual(planes[0, 3, 5], 1)  # e8
        # two white knights, a3, g1
        self.assertEqual(np.sum(planes[:, :, 7]), 2)
        self.assertEqual(planes[5, 7, 7], 1)
        self.assertEqual(planes[7, 1, 7], 1)

        # color to play is black
        self.assertEqual(np.sum(features[:, :, 112]), 0)
        # total moves is 1
        self.assertEqual(np.sum(features[:, :, 113]), 64)
        # all castling is enabled
        self.assertEqual(np.sum(features[:, :, 114:118]), 64*4)
        # non process moves is 0. The number is increased after black move
        self.assertEqual(np.sum(features[:, :, 118]), 0)
