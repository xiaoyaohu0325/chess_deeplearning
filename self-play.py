from policy import ResnetPolicy
from timeit import default_timer as timer
from preprocessing import game_converter
import logging
import chess
import argparse
import os

# from player.MCTSPlayer import MCTSPlayerMixin
# from player.RandomPlayer import RandomPlayerMixin
from player.Node import Node
from tree_exporter import export_node
import logging
import daiquiri
import pyximport
pyximport.install()

from player.MCTSPlayer_C import *

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


MAX_MOVES = 300


def play_games(model, weights, out_dir, games, pid):
    policy = ResnetPolicy.load_model(model)
    policy.model.load_weights(weights)

    for i in range(games):
        # start a new
        start = timer()
        root_node = Node()
        mctc = MCTSPlayerMixin(policy, 300)
        # random_player = RandomPlayerMixin()

        next_node = root_node
        moves = 0
        while True:
            # start_search = timer()
            move, win_rate = mctc.suggest_move(next_node)
            # end_search = timer()

            # g = export_node(next_node, show_details=False)
            # g.render(filename=str(moves), directory=out_dir)

            next_node = next_node.children[move]
            moves += 1
            # print('search move ', end_search - start_search, "win_rate:", win_rate)
            if moves > MAX_MOVES or next_node.board.is_game_over():
                break

        if moves > MAX_MOVES:
            next_node.feed_back_winner(force=True)
        else:
            next_node.feed_back_winner()

        end = timer()
        result = next_node.board.result()
        print("game ", i, " finished!  elapsed ", end - start, ", round: ", next_node.n,
              ", result:", result)

        game_converter.save_pgn_to_hd5(file_path=os.path.join(out_dir, "pgn_{0}.h5".format(str(pid))),
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path=os.path.join(out_dir, "features_{0}.h5".format(str(pid))),
                                       game_tree=root_node)

# def search_move(s0_node, n_simulation, n_depth):
#     for i in range(n_simulation):
#         # step 1: select to time step L
#         selected_node = s0_node.select(depth=n_depth)
#         # step 2: expand an evaluate
#         reward = selected_node.evaluate()
#         # step 3: backup
#         selected_node.update_recursive(reward, s0_node.depth, selected_node.board.turn)


def run_self_play(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform self play to generate data.')
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("weights", help="Path to a weights file")
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    parser.add_argument("--games", "-n", help="Number of games to generate. Default: 1", type=int,
                        default=1)
    parser.add_argument("--pid", "-p", help="unique id of the generated h5 file. Default: 0", type=int,
                        default=0)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)
    play_games(args.model, args.weights, args.out_directory, args.games, args.pid)


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # logging.basicConfig(filename='./out/predict.log', level=logging.INFO)
    run_self_play()
