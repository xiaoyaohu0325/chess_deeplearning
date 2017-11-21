from game import Game
from player import AIPlayer
from policy import ResnetPolicy
import argparse
import chess


def play(white_player, black_player, n_game=100):
    for i in range(n_game):
        game = Game(white_player, black_player)
        move = game.play()
        while move is not None:
            move = game.play()

        print('game', i, 'finished! Full moves', game.fullmove_count(), 'winner:', game.winner_color())
        game.save_to_h5("./out/checkpoint/gen0_gen1.h5")


def run_checkpoint(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform training on a policy network.')
    # required args
    parser.add_argument("model_1", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("weights_1", help="A .h5 file of training data")
    parser.add_argument("model_2", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("weights_2", help="A .h5 file of training data")
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--simulations", "-s", help="Simulation numbers. Default: 100", type=int,
                        default=100)  # noqa: E501
    parser.add_argument("--depth", "-d", help="Search depth of simulation. Default: 1", type=int,
                        default=1)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    policy_1 = ResnetPolicy.load_model("./out/model/model_2_128.json")
    policy_1.model.load_weights("./out/model/random_weights_2_128")
    policy_2 = ResnetPolicy.load_model("./out/model/model_2_128.json")
    policy_2.model.load_weights("./out/train/weights.00002.hdf5")

    # assign black and white
    p1 = AIPlayer('gen0', chess.WHITE, policy_1, simulation=400, depth=1)
    p2 = AIPlayer('gen1', chess.BLACK, policy_2, simulation=400, depth=1)
    play(p1, p2)

    # switch color
    p1 = AIPlayer('gen0', chess.BLACK, policy_1, simulation=400, depth=1)
    p2 = AIPlayer('gen1', chess.WHITE, policy_2, simulation=400, depth=1)
    play(p2, p1)


if __name__ == '__main__':
    run_checkpoint()
