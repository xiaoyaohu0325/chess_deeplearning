import argparse
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from config import FLAGS, HPS
from collections import namedtuple

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)


@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info("{0}: {1:.3f} seconds".format(message, (tock - tick)))


'''
params:
    @ train_step: total number of mini-batch updates
    @ usage: learning rate annealling
'''


def schedule_lrn_rate(train_step):
    """train_step equals total number of min_batch updates"""
    f = 1  # rl schedule factor
    lr = 0.2
    if train_step < 1 * f:
        lr = 0.02
    elif train_step < 2 * f:
        lr = 0.002
    else:
        lr = 0.0002
    return lr


'''
params:
    @ usage: self play with search pipeline
'''


def selfplay(flags=FLAGS, hps=HPS):
    # from utils.load_data_sets import DataSet
    from model.self_play_worker import SelfPlayWorker
    from model.network import Network

    # test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))
    # test_dataset = None

    """set the batch size to -1==None"""
    flags.n_batch = -1
    net = Network(flags, hps)
    worker = SelfPlayWorker(net, flags)

    def selfplay_train(epoch: int):
        lr = schedule_lrn_rate(epoch)
        worker.run(lr=lr)

    # # TODO: consider tensorflow copy_to_graph
    # def get_best_model():
    #     return Network(flags, hps)

    # def evaluate_generations():
    #     best_model = get_best_model()
    #     worker.evaluate_model(best_model)
    #
    # def evaluate_testset():
    #     worker.evaluate_testset(test_dataset)

    """Self Play Pipeline starts here"""
    for g_epoch in range(flags.global_epoch):
        logger.info('Global epoch {0} start.'.format(g_epoch))

        """Train"""
        selfplay_train(g_epoch)

        """Evaluate on test dataset"""
        # evaluate_testset()

        """Evaluate against best model"""
        # evaluate_generations()

        logger.info('Global epoch {0} finish.'.format(g_epoch))


'''
params:
    @ usage: train a supervised learning network
'''


def train(flags=FLAGS, hps=HPS):
    # from utils.load_data_sets import DataSet
    from model.network import Network

    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    net = Network(flags, hps)

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    train_chunk_files = [os.path.join(flags.processed_dir, fname)
                         for fname in os.listdir(flags.processed_dir)
                         if TRAINING_CHUNK_RE.match(fname)]

    def training_datasets():
        random.shuffle(train_chunk_files)
        return (DataSet.read(file) for file in train_chunk_files)

    global_step = 0
    lr = flags.lr

    with open("result.txt", "a") as f:
        for g_epoch in range(flags.global_epoch):

            """Train"""
            lr = schedule_lrn_rate(g_epoch)
            for train_dataset in training_datasets():
                global_step += 1
                # prepare training set
                logger.info("Global step {0} start".format(global_step))
                train_dataset.shuffle()
                with timer("training"):
                    net.train(train_dataset, lrn_rate=lr)

                """Evaluate"""
                if global_step % 1 == 0:
                    with timer("test set evaluation"):
                        net.test(test_dataset, proportion=0.25,
                                 force_save_model=global_step % 10 == 0)

                logger.info('Global step {0} finshed.'.format(global_step))
            logger.info('Global epoch {0} finshed.'.format(g_epoch))


'''
params:
    @ usage: test a trained network on test dataset
'''


def test(flags=FLAGS, hps=HPS):
    # from utils.load_data_sets import DataSet
    from model.network import Network
    import tensorflow as tf

    net = Network(flags, hps)

    # print(net.sess.run({var.name:var for var in tf.global_variables() if 'bn' in var.name}))

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    with timer("test set evaluation"):
        net.test(test_dataset, proportion=0.25, force_save_model=False)


if __name__ == '__main__':

    if not os.path.exists('../out/train_log'):
        os.makedirs('../out/train_log')

    if not os.path.exists('../out/test_log'):
        os.makedirs('../out/test_log')

    if not os.path.exists('../out/savedmodels'):
        os.makedirs('../out/savedmodels')

    if not os.path.exists('../out/result.txt'):
        # hacky way to creat a file
        open("result.txt", "a").close()

    fn = {'train': lambda: train(),
          'selfplay': lambda: selfplay(),
          'test': lambda: test()}

    if fn.get(FLAGS.MODE, 0) != 0:
        fn[FLAGS.MODE]()
    else:
        logger.info('Please choose a mode among "train", "selfplay", "gtp", and "test".')
