import unittest
import chess
import numpy as np
from util.features import extrac_piece_planes


class TestFeatures(unittest.TestCase):
    def test_extract_piece_planes(self):
        board = chess.Board(fen="1n2kb2/4p3/b1p3rp/5PP1/pP1p2P1/N2n3P/PB1P2K1/1r2QBNR w - - 22 51")
        planes = extrac_piece_planes(board)
        self.assertEqual(planes.shape, (8, 8, 12))
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
