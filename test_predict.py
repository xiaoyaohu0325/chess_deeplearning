import numpy as np
from timeit import default_timer as timer
import unittest
from policy import ResnetPolicy
from preprocessing import game_converter
from game_tree import TreeNode
from tree_exporter import export_node
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
        policy = ResnetPolicy.load_model("./out/model/model_4_128.json")
        policy.model.load_weights('./out/train/10_128/weights.00008.hdf5')
        fen_start = "r1bqkbr1/ppp1pppp/n4n2/8/1P1pP3/2NQ4/P1PP1PPP/R1B1KBNR w KQq - 0 6"
        fen_middle = "1n2kb2/4p3/b1p3rp/5PP1/pP1p2P1/N2n3P/PB1P2K1/1r2QBNR w - - 22 51"
        fen_end = "3kq3/3rp1p1/8/6p1/4B3/2p3P1/2KP1P1N/RN5R w - - 84 76"
        root = TreeNode(None, policy=policy, fen=fen_middle)
        root.depth = 102

        # for item in root.children.items():
        #     action = item[0]
        #     print("action ", action, ", move ", item[1].move)
        #     print("from ", chess.square_name(action[0]), ", to ", chess.square_name(action[1]))
        start = timer()
        for i in range(400):
            selected_node = root.select(depth=3)
            reward = selected_node.evaluate()
            # (action, sub_node) = max(root.children.items(), key=lambda act_node: act_node[1].get_value())
            # from_name = chess.square_name(action[0])
            # to_name = chess.square_name(action[1])
            # move_name = from_name + to_name
            # logging.info("iteration %d", i)
            # logging.info("selected move: " + move_name)
            # logging.info("value {0:.4f}".format(sub_node.get_value()))
            # self._print_node_info(root)
            g = export_node(root)
            g.render(filename=str(i), directory='./out/view/middle')

            selected_node.update_recursive(reward, root.depth, selected_node.board.turn)

        end = timer()
        print("predict elapsed ", end - start)

    def _print_node_info(self, node):
        for (action, subnode) in node.children.items():
            from_square_name = chess.square_name(action[0])
            to_square_name = chess.square_name(action[1])
            logging.info("move: %s%s, value: %s", from_square_name, to_square_name, subnode._weights())
