from game import Game
from player import AIPlayer, RandomPlayer
from policy import ResnetPolicy
import argparse
import chess
import os


def play(white_player, black_player, out_file, n_game=100):
    for i in range(n_game):
        game = Game(white_player, black_player)
        move = game.play()
        while move is not None:
            move = game.play()

        print('game', i, 'finished! Full moves', game.fullmove_count(), 'winner:', game.winner_color())
        game.save_to_h5(out_file)


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
                        default=100)  # noqa: E501
    parser.add_argument("--depth", "-d", help="Search depth of simulation. Default: 1", type=int,
                        default=1)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # assign black and white
    p1 = _create_player(chess.WHITE, args.model_1, args.weights_1,
                        args.simulations, args.depth)
    p2 = _create_player(chess.BLACK, args.model_2, args.weights_2,
                        args.simulations, args.depth)
    play(p1, p2, args.out_file)

    # switch color
    p1 = _create_player(chess.BLACK, args.model_1, args.weights_1,
                        args.simulations, args.depth)
    p2 = _create_player(chess.WHITE, args.model_2, args.weights_2,
                        args.simulations, args.depth)
    play(p2, p1, args.out_file)


def _create_player(color, model, weights, simulation=400, depth=1):
    if model is None or weights is None:
        return RandomPlayer('random', color)
    else:
        policy = ResnetPolicy.load_model(model)
        model_name = os.path.basename(model)
        name = os.path.splitext(model_name)[0]
        policy.model.load_weights(weights)
        return AIPlayer(name, color, policy, simulation, depth)


if __name__ == '__main__':
    run_checkpoint()
