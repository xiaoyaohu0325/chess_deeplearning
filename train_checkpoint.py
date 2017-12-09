from game import Game
from player.MCTSPlayer import MCTSPlayerMixin
from player.RandomPlayer import RandomPlayerMixin
from policy import ResnetPolicy
import argparse
import os
import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def play(white_player, black_player, out_file, n_game=100):
    for i in range(n_game):
        game = Game(white_player, black_player)
        move = game.play()
        while move is not None:
            move = game.play()

        print('game', i, 'finished! Full moves', game.fullmove_count(), 'winner:', game.winner_color())
        game.save_pgn(out_file)


def run_checkpoint(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform training on a policy network.')
    # required args
    parser.add_argument("--model_1", "-m1", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("--weights_1", "-w1", help="A .h5 file of training data")
    parser.add_argument("--model_2", "-m2", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("--weights_2", "-w2", help="A .h5 file of training data")
    parser.add_argument("--out_file", "-o", help="output file which the pgn will save")
    # frequently used args
    parser.add_argument("--simulations", "-s", help="Simulation numbers. Default: 100", type=int,
                        default=400)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # assign black and white
    p1 = _create_player(args.model_1, args.weights_1,  args.simulations)
    p2 = _create_player(args.model_2, args.weights_2, args.simulations)
    play(p1, p2, args.out_file)

    # switch color
    play(p2, p1, args.out_file)


def _create_player(model, weights, simulation=400):
    if model is None or weights is None:
        return RandomPlayerMixin(name='random')
    else:
        policy = ResnetPolicy.load_model(model)
        model_name = os.path.basename(model)
        name = os.path.splitext(model_name)[0]
        policy.model.load_weights(weights)
        return MCTSPlayerMixin(policy, simulation, name=name)


if __name__ == '__main__':
    run_checkpoint()
