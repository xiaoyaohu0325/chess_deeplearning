import unittest
from timeit import default_timer as timer
from policy import ResnetPolicy
from player.MCTSPlayer import MCTSPlayerMixin
from player.Node import Node
from game import Game
# from tree_exporter import export_node
import logging
import daiquiri
# import pyximport
# pyximport.install()
#
# from player.MCTSPlayer_C import *

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class MCTSTest(unittest.TestCase):

    def test_suggest_move(self):
        policy = ResnetPolicy.load_model('../out/model/model_alphazero.json')
        policy.model.load_weights('../out/model/weights_alphazero.h5')

        mc_root = MCTSPlayerMixin(policy, 300)
        game = Game(mc_root, mc_root)
        start = timer()
        move, win_rate = game.play()
        end = timer()

        print("suggest move elapse:", end-start)
        print("selected move:", move.uci(), ', win_rate:', win_rate)
        # g = export_node(root_node)
        # g.render(filename=str(0), directory='../out/view/mcts')
