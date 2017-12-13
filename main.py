import argparse
from time import time
from contextlib import contextmanager
import os
import sys
from self_play_worker_keras import SelfPlayWorker
from policy import ResnetPolicy

import logging
import daiquiri
daiquiri.setup(level=logging.INFO)
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


def selfplay(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform self play to generate data.')
    # required args
    parser.add_argument("--model", "-m", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("--weights", "-w", help="Path to a weights file")
    parser.add_argument("--min_steps", "-s", help="unique id of the generated h5 file. Default: 0", type=int,
                        default=0)
    parser.add_argument("--max_steps", "-x", help="unique id of the generated h5 file. Default: 0", type=int,
                        default=200000)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    policy = ResnetPolicy.load_model(args.model, args.min_steps)
    policy.model.load_weights(args.weights)

    worker = SelfPlayWorker(policy, os.path.join(_PATH_, 'out/train'), simulation=200, batch_size=4096)
    worker.run(args.min_steps, args.max_steps)


if __name__ == '__main__':
    selfplay()
