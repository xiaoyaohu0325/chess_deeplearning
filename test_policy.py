import unittest
from policy import ResnetPolicy
from keras.utils import plot_model


class TestCNNPolicy(unittest.TestCase):

    def test_save_model(self):
        policy = ResnetPolicy(
            init_network=True,
            residual_blocks=16)
        policy.save_model(r"./out/model/128/depth-16/model_16_128.json")

    def test_plot(self):
        policy = ResnetPolicy.load_model(r"./out/model/128/depth-16/model_16_128.json")
        plot_model(policy.model,
                   show_shapes=True,
                   to_file=r"./out/model/128/depth-16/model_16_128.png")
