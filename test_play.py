import unittest
from timeit import default_timer as timer
from random import randint
from game_tree import TreeNode


class TestGameTree(unittest.TestCase):

    def test_play(self):
        root_node = TreeNode(None)
        current_node = root_node

        start = timer()
        print("game start")
        while True:
            if current_node.board.is_game_over(claim_draw=True):
                print("result: ", current_node.board.result(claim_draw=True))
                break
            start_t = timer()
            moves = []
            for legal_move in current_node.board.generate_legal_moves():
                moves.append(legal_move)
            end_t = timer()
            print("get_legal_moves elapse: ", end_t - start_t, " available moves: ", len(moves))
            if len(moves) > 0:
                move = moves[randint(0, len(moves) - 1)]
                action = (move.from_square, move.to_square)
                start_t = timer()
                current_node = current_node.append_move(action)
                end_t = timer()
                print("append node elapse: ", end_t - start_t, ", move: ", current_node.move)

        end = timer()
        print("tree depth: ", current_node.depth)
        print("game over, elapsed ", end - start)
        root_node.save_as_pgn(r"./out/self-play.pgn")

