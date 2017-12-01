import unittest
from timeit import default_timer as timer
from policy import ResnetPolicy
# from player.MCTSPlayer import MCTSPlayerMixin
from player.Node import Node
from util.features import action_to_icu
from tree_exporter import export_node
import logging
import daiquiri
import pyximport
pyximport.install()

from player.MCTSPlayer_C import *

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class MCTSTest(unittest.TestCase):

    def test_suggest_move(self):
        policy = ResnetPolicy.load_model('../out/model/128/model_4_128.json')
        policy.model.load_weights('../out/model/128/random_weights_4_128')

        root_node = Node()
        mc_root = MCTSPlayerMixin(policy, 300)
        start = timer()
        move, win_rate = mc_root.suggest_move(root_node)
        end = timer()

        print("suggest move elapse:", end-start)
        print("selected move:", action_to_icu(move), ', win_rate:', win_rate)
        g = export_node(root_node)
        g.render(filename=str(0), directory='../out/view/mcts')
