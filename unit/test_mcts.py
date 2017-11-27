import unittest
from timeit import default_timer as timer
from policy import ResnetPolicy
from player.MCTSPlayer import MCTSPlayerMixin
from player.Position import Position
from util.features import action_to_icu


class MCTSTest(unittest.TestCase):

    def test_suggest_move(self):
        policy = ResnetPolicy.load_model('../out/model/128/model_10_128.json')
        policy.model.load_weights('../out/model/128/random_weights_10_128')

        position = Position()
        mc_root = MCTSPlayerMixin(policy, 800)
        start = timer()
        move, win_rate = mc_root.suggest_move(position)
        end = timer()

        print("suggest move elapse:", end-start)
        print("selected move:", action_to_icu(move), ', win_rate:', win_rate)
