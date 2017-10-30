import numpy as np
from timeit import default_timer as timer
import unittest
from policy import ResnetPolicy
from preprocessing import game_converter


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
