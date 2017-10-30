import unittest
from policy import ResnetPolicy
from keras.utils import plot_model


class TestCNNPolicy(unittest.TestCase):

    def test_save_model(self):
        policy = ResnetPolicy(init_network=True)
        policy.save_model(r"./out/model.json", r"./out/random_weights")

    def test_plot(self):
        policy = ResnetPolicy.load_model(r"./out/model.json")
        plot_model(policy.model,
                   show_shapes=True,
                   to_file=r"./out/model.png")
