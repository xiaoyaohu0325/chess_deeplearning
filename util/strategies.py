import random
import numpy as np
import daiquiri
import logging
from util.features import index_to_action

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def sorted_moves(probability_array):
    return np.argsort(-probability_array)


def select_most_likely(position, move_probabilities):
    """Select the most resonable move according to sorted probability"""
    for idx in sorted_moves(move_probabilities):
        action = index_to_action(idx)
        if position.is_legal_move(action):
            return action
    return None


def select_weighted_random(position, move_probabilities):
    """select move according to their relative probability"""
    selection = random.random()
    cdf = move_probabilities.cumsum()
    selected_move = index_to_action(cdf.searchsorted(selection, side="right"))

    if position.is_legal_move(selected_move):
        return selected_move
    else:
        # inexpensive fallback in case an illegal move is chosen.
        return select_most_likely(position, move_probabilities)


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
