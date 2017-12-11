import logging
import daiquiri
from contextlib import contextmanager
from time import time
import os
import h5py

from game import Game
from player.MCTSPlayer import MCTSPlayerMixin

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info("{0}: {1:.3f} seconds".format(message, (tock - tick)))


class SelfPlayWorker(object):

    def __init__(self, net, work_dir, games):
        self.net = net
        self.player = MCTSPlayerMixin(net, 800)
        self.max_moves = 500

        self.n_moves_per_train = 4096
        self.n_games = games
        self.out_dir = work_dir

    def _start_index_of_features(self):
        if not os.path.exists(os.path.join(self.out_dir, "features.h5")):
            return 0
        try:
            h5f = h5py.File(os.path.join(self.out_dir, "features.h5"))
            return len(h5f["features"])
        finally:
            h5f.close()

    def _extract_training_data(self, start_index):
        try:
            h5f = h5py.File(os.path.join(self.out_dir, "features.h5"))
            feature_dataset = h5f["features"]
            pi_dataset = h5f["pi"]
            rewards_dataset = h5f["rewards"]
            return (feature_dataset[start_index:start_index+self.n_moves_per_train],
                    pi_dataset[start_index:start_index+self.n_moves_per_train],
                    rewards_dataset[start_index:start_index+self.n_moves_per_train])
        finally:
            h5f.close()

    def run(self, step):
        moves_counter = 0
        start_index = self._start_index_of_features()
        training_steps = step

        for i in range(self.n_games):
            """self play with MCTS search"""

            with timer("Self-Play Simulation Game #{0}".format(i+1)):
                game = Game(self.player, self.player)
                game.play_to_end(max_moves=self.max_moves)
                game.save_pgn(os.path.join(self.out_dir, "pgn.h5"))
                from_index, to_index = game.save_features(os.path.join(self.out_dir, "features.h5"))

            moves_counter += (to_index - from_index)

            if moves_counter >= self.n_moves_per_train:
                training_samples = self._extract_training_data(start_index)
                start_index += self.n_moves_per_train
                self.net.train(training_samples, training_steps, self.out_dir)
                # features left are moved to next train
                moves_counter = to_index - start_index
                training_steps += 1
