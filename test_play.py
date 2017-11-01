import unittest
from timeit import default_timer as timer
from random import randint
from game_tree import TreeNode
from preprocessing import game_converter


class TestGameTree(unittest.TestCase):

    def test_play(self):
        for i in range(10):
            print("iteration ", i)
            root_node = TreeNode(None)
            current_node = root_node

            # start = timer()
            # print("game start")
            while True:
                if current_node.board.is_game_over(claim_draw=True):
                    # print("result: ", current_node.board.result(claim_draw=True))
                    break
                # start_t = timer()
                moves = []
                for legal_move in current_node.board.generate_legal_moves():
                    moves.append(legal_move)
                # end_t = timer()
                # print("get_legal_moves elapse: ", end_t - start_t, " available moves: ", len(moves))
                if len(moves) > 0:
                    move = moves[randint(0, len(moves) - 1)]
                    action = (move.from_square, move.to_square)
                    # start_t = timer()
                    current_node = current_node.append_move(action)
                    # end_t = timer()
                    # print("append node elapse: ", end_t - start_t, ", move: ", current_node.move)

            # end = timer()
            # print("tree depth: ", current_node.depth)
            # print("game over, elapsed ", end - start)
            # root_node.save_as_pgn(r"./out/self-play-not-draw.pgn")
            game_converter.save_pgn_to_hd5(r"./out/self-play-1.hdf5",
                                           current_node.export_pgn_str(),
                                           current_node.board.result(claim_draw=True))

    def test_white_win_reward(self):
        i = 0
        while True:  # choose a game that there is a winner
            print("iteration ", i)
            i += 1
            root_node = TreeNode(None)
            current_node = root_node

            # start = timer()
            # print("game start")
            while True:
                if current_node.board.is_game_over(claim_draw=True):
                    # print("result: ", current_node.board.result(claim_draw=True))
                    break
                # start_t = timer()
                moves = []
                for legal_move in current_node.board.generate_legal_moves():
                    moves.append(legal_move)
                # end_t = timer()
                # print("get_legal_moves elapse: ", end_t - start_t, " available moves: ", len(moves))
                if len(moves) > 0:
                    move = moves[randint(0, len(moves) - 1)]
                    action = (move.from_square, move.to_square)
                    # start_t = timer()
                    current_node = current_node.append_move(action)
                    # end_t = timer()
                    # print("append node elapse: ", end_t - start_t, ", move: ", current_node.move)

            result = current_node.board.result(claim_draw=True)
            if result == "1-0":
                current_node.feed_back_winner()
                print("result: ", result)
                node = root_node.next_node()
                while node is not None:
                    print("turn ", node.board.turn, ", reward: ", node.reward)
                    node = node.next_node()
                break

    def test_black_win_reward(self):
        i = 0
        while True:  # choose a game that there is a winner
            print("iteration ", i)
            i += 1
            root_node = TreeNode(None)
            current_node = root_node

            # start = timer()
            # print("game start")
            while True:
                if current_node.board.is_game_over(claim_draw=True):
                    # print("result: ", current_node.board.result(claim_draw=True))
                    break
                # start_t = timer()
                moves = []
                for legal_move in current_node.board.generate_legal_moves():
                    moves.append(legal_move)
                # end_t = timer()
                # print("get_legal_moves elapse: ", end_t - start_t, " available moves: ", len(moves))
                if len(moves) > 0:
                    move = moves[randint(0, len(moves) - 1)]
                    action = (move.from_square, move.to_square)
                    # start_t = timer()
                    current_node = current_node.append_move(action)
                    # end_t = timer()
                    # print("append node elapse: ", end_t - start_t, ", move: ", current_node.move)

            result = current_node.board.result(claim_draw=True)
            if result == "0-1":
                current_node.feed_back_winner()
                print("result: ", result)
                node = root_node.next_node()
                while node is not None:
                    print("turn ", node.board.turn, ", reward: ", node.reward)
                    node = node.next_node()
                break

    def test_draw_reward(self):
        i = 0
        while True:  # choose a game that there is a winner
            print("iteration ", i)
            i += 1
            root_node = TreeNode(None)
            current_node = root_node

            # start = timer()
            # print("game start")
            while True:
                if current_node.board.is_game_over(claim_draw=True):
                    # print("result: ", current_node.board.result(claim_draw=True))
                    break
                # start_t = timer()
                moves = []
                for legal_move in current_node.board.generate_legal_moves():
                    moves.append(legal_move)
                # end_t = timer()
                # print("get_legal_moves elapse: ", end_t - start_t, " available moves: ", len(moves))
                if len(moves) > 0:
                    move = moves[randint(0, len(moves) - 1)]
                    action = (move.from_square, move.to_square)
                    # start_t = timer()
                    current_node = current_node.append_move(action)
                    # end_t = timer()
                    # print("append node elapse: ", end_t - start_t, ", move: ", current_node.move)

            result = current_node.board.result(claim_draw=True)
            if result == "1/2-1/2":
                current_node.feed_back_winner()
                print("result: ", result)
                node = root_node.next_node()
                while node is not None:
                    print("turn ", node.board.turn, ", reward: ", node.reward)
                    node = node.next_node()
                break
