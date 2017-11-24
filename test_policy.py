import unittest
from policy import ResnetPolicy
from keras.utils import plot_model


class TestCNNPolicy(unittest.TestCase):

    def test_save_model(self):
        policy = ResnetPolicy(
            init_network=True,
            residual_blocks=10)
        policy.save_model(r"./out/model/1024/model_10_128.json", r"./out/model/1024/random_weights_10_128")

    def test_plot(self):
        policy = ResnetPolicy.load_model(r"./out/model/1024/model_10_128.json")
        plot_model(policy.model,
                   show_shapes=True,
                   to_file=r"./out/model/1024/model_10_128.png")
