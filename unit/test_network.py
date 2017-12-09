from config import FLAGS, HPS
import unittest
from model.network import Network

class NetworkTest(unittest.TestCase):

    def test_save(self):
        network = Network(FLAGS, HPS)
        network.save_model("../out/tf", 1)
