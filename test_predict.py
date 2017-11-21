import numpy as np
from timeit import default_timer as timer
import unittest
from policy import ResnetPolicy
from preprocessing import game_converter
from game_tree import TreeNode
import chess
import logging

logging.basicConfig(filename='./out/predict.log', level=logging.INFO)


class TestPredict(unittest.TestCase):

    def test_predict(self):
        policy = ResnetPolicy.load_model("./out/model/model_2_128.json")

        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        to_predict = game_converter.fen_to_features(fen)
        print(to_predict.shape)
        # for i in range(18):
        #     print(to_predict[:, :, i])

        start = timer()
        output = policy.forward([to_predict])
        end = timer()
        print("predict elapsed ", end - start)
        print("output length: ", len(output))
        print(output[0].shape)  # (1, 362)
        print(output[1].shape)  # (1, 1)

        prior_p = np.reshape(output[0][0], (7, 64))
        print(prior_p[0])
        print(np.sum(prior_p[0]))

    def test_search(self):
        policy = ResnetPolicy.load_model("./out/model/model_2_128.json")
        root = TreeNode(None, policy)

        # for item in root.children.items():
        #     action = item[0]
        #     print("action ", action, ", move ", item[1].move)
        #     print("from ", chess.square_name(action[0]), ", to ", chess.square_name(action[1]))
        start = timer()
        for i in range(800):
            selected_node = root.select(depth=3)
            reward = selected_node.evaluate()
            selected_node.update_recursive(reward, 0, selected_node.board.turn)
            (action, sub_node) = max(root.children.items(), key=lambda act_node: act_node[1].get_value())
            from_name = chess.square_name(action[0])
            to_name = chess.square_name(action[1])
            move_name = from_name + to_name
            logging.info("iteration %d", i)
            logging.info("selected move: " + move_name)
            logging.info("value {0:.4f}".format(sub_node.get_value()))
            self._print_node_info(root)

        end = timer()
        print("predict elapsed ", end - start)

    def _print_node_info(self, node):
        for (action, subnode) in node.children.items():
            from_square_name = chess.square_name(action[0])
            to_square_name = chess.square_name(action[1])
            logging.info("move: %s%s, value: %s", from_square_name, to_square_name, subnode._weights())
