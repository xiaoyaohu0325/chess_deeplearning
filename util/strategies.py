import random
import numpy as np
import daiquiri
import logging
from util.features import index_to_action

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

def sorted_moves(probability_array):
    return np.argsort(-probability_array)

# def simulate_game_mcts(policy, position, playouts=1600):
#     """Simulates a game starting from a position, using a policy network"""
#
#     mc_root = MCTSPlayerMixin(policy, playouts)
#
#     while not position.board.is_game_over():
#         move = mc_root.suggest_move(position)
#         position.play_move(move, mutate=True, move_prob=mc_root.move_prob(
#             key=None, position=position))
#         logger.debug('Move at step {0} is {1}'.format(position.n, move))
#
#     # return game result and stats
#     return position
