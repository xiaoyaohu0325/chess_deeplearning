import numpy as np
from timeit import default_timer as timer
import unittest
from policy import ResnetPolicy
from preprocessing import game_converter
from game_tree import TreeNode
import chess
import logging

logging.basicConfig(filename='./out/predict.log', level=logging.DEBUG)


class TestPredict(unittest.TestCase):

    def test_predict(self):
        policy = ResnetPolicy.load_model("./out/model.json")

        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        to_predict = game_converter.fen_to_features(fen)
        print(to_predict.shape)
        for i in range(18):
            print(to_predict[:, :, i])

        start = timer()
        output = policy.forward([to_predict])
        end = timer()
        print("predict elapsed ", end - start)
        print("output length: ", len(output))
        print(output[0].shape)  # (1, 362)
        print(output[1].shape)  # (1, 1)

    def test_search(self):
        policy = ResnetPolicy.load_model("./out/model/model_2_128.json")
        root = TreeNode(None, policy)

        # for item in root.children.items():
        #     action = item[0]
        #     print("action ", action, ", move ", item[1].move)
        #     print("from ", chess.square_name(action[0]), ", to ", chess.square_name(action[1]))
        start = timer()
        for i in range(800):
            selected_node = root.select(depth=2)
            reward = selected_node.evaluate()
            selected_node.update_recursive(reward)

        end = timer()
        print("predict elapsed ", end - start)

        root.play()
        size = len(root.pi_from)
        for i in range(size):
            if root.pi_from[i] > 0:
                for j in range(size):
                    if root.pi_to[j] > 0:
                        print("from:", chess.square_name(i), ", to:", chess.square_name(j),
                              "from_pi:", root.pi_from[i],
                              "to_pi:", root.pi_to[j])
