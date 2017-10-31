import unittest
from game_tree import TreeNode
import chess


class TestGameTree(unittest.TestCase):
    def test_promotion(self):
        root = TreeNode(parent=None, fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        child = TreeNode(parent=root, action=(48, 56))

        self.assertTrue(child.move.promotion)
        self.assertEqual('a7', chess.square_name(child.move.from_square))
        self.assertEqual('a8', chess.square_name(child.move.to_square))
