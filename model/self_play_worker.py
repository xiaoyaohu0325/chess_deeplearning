import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from contextlib import contextmanager
from time import time


@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info("{0}: {1:.3f} seconds".format(message, (tock - tick)))


class SelfPlayWorker(object):

    def __init__(self, net, rl_flags):
        self.net = net

        self.N_moves_per_train = rl_flags.N_moves_per_train
        self.N_games = rl_flags.selfplay_games_per_epoch
        self.playouts = rl_flags.num_playouts

        # self.position = go.Position(to_play=go.BLACK)
        # self.final_position_collections = []

        self.num_games_to_evaluate = rl_flags.selfplay_games_against_best_model

    # def reset_position(self):
    #     # self.position = go.Position(to_play=go.BLACK)
    #     pass

    '''
    params:
        @ lr: learning rate, controled by outer loop
        usage: run self play with search
    '''

    def run(self, lr=0.01):

        moves_counter = 0

        for i in range(self.N_games):
            """self play with MCTS search"""

            with timer("Self-Play Simulation Game #{0}".format(i+1)):
                final_position, agent_resigned, false_positive = simulate_game_mcts(self.net, self.position,
                                                                                    playouts=self.playouts, resignThreshold=self.resign_threshold, no_resign=self.no_resign_this_game)

                logger.debug('Game #{0} Final Position:\n{1}'.format(i+1, final_position))

            # reset game board
            # self.reset_position()

            # Discard game that resign too early
            # if final_position.n <= self.dicard_game_threshold:
            #     logger.debug('Game #{0} ends too early, discard!'.format(i+1))
            #     continue

            # add final_position to history
            self.final_position_collections.append(final_position)
            moves_counter += final_position.n

            if moves_counter >= self.N_moves_per_train:
                winners_training_samples, losers_training_samples = extract_moves(
                    self.final_position_collections)
                self.net.train(winners_training_samples, direction=1., lrn_rate=lr)
                # self.net.train(losers_training_samples, direction=-1., lrn_rate=lr)
                # self.final_position_collections = []
                moves_counter = 0

    # def evaluate_model(self, best_model):
    #     self.reset_position()
    #     final_positions = simulate_many_games(
    #         self.net, best_model, [self.position] * self.num_games_to_evaluate)
    #     win_ratio = get_winrate(final_positions)
    #     if win_ratio < 0.55:
    #         logger.info(f'Previous Generation win by {win_ratio:.4f}% the game!')
    #         self.net.close()
    #         self.net = best_model
    #     else:
    #         logger.info(f'Current Generation win by {win_ratio:.4f}% the game!')
    #         best_model.close()
    #         # self.net.save_model(name=round(win_ratio,4))
    #     self.reset_position()
    #
    # def evaluate_testset(self, test_dataset):
    #     with timer("test set evaluation"):
    #         self.net.test(test_dataset, proportion=.1, no_save=True)
