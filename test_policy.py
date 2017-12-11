import unittest
from policy import ResnetPolicy
from keras.utils import plot_model


class TestCNNPolicy(unittest.TestCase):

    def test_save_model(self):
        policy = ResnetPolicy(
            init_network=True)
        policy.save_model(r"./out/model/model_alphazero.json", r"./out/model/weights_alphazero.h5")

    def test_plot(self):
        policy = ResnetPolicy.load_model(r"./out/model/model_alphazero.json")
        plot_model(policy.model,
                   show_shapes=True,
                   to_file=r"./out/model/model_alphazero.png")

    def test_save_weights(self):
        policy = ResnetPolicy.load_model(r"./out/model/model_alphazero.json")
        policy.save_weights(r"./out/model/weights_alphazero.h5")
