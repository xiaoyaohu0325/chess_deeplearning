import unittest
from policy import ResnetPolicy
from keras.utils import plot_model


class TestCNNPolicy(unittest.TestCase):

    def test_save_model(self):
        policy = ResnetPolicy(
            init_network=True,
            residual_blocks=19)
        policy.save_model(r"./out/model/model_19_128.json", r"./out/model/random_weights_19_128")

    # def test_plot(self):
    #     policy = ResnetPolicy.load_model(r"./out/model/model_2_128.json")
    #     plot_model(policy.model,
    #                show_shapes=True,
    #                to_file=r"./out/model/model_2_128.png")
