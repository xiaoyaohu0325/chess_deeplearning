from policy import ResnetPolicy
from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter
import logging
import chess
import argparse
import os
# from tree_exporter import export_node


def play_games(model, weights, out_dir, games, pid, simulations, depth):
    policy = ResnetPolicy.load_model(model)
    policy.model.load_weights(weights)

    for i in range(games):
        # start a new
        start = timer()
        root_node = TreeNode(None, policy=policy)
        next_node = root_node
        moves = 0
        while True:
            start_search = timer()
            search_move(next_node, simulations, depth)
            end_search = timer()

            # g = export_node(next_node, expand=False)
            # g.render(filename=str(moves), directory='./out/view/3_300')

            next_node = next_node.play()
            moves += 1
            print('search move ', end_search - start_search)
            if next_node.board.is_game_over(claim_draw=True):
                break

        next_node.feed_back_winner()

        game_converter.save_pgn_to_hd5(file_path=os.path.join(out_dir, "pgn_{0}.h5".format(str(pid))),
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path=os.path.join(out_dir, "features_{0}.h5".format(str(pid))),
                                       game_tree=root_node)
        end = timer()
        print("game ", i, " finished!  elapsed ", end-start, ", round: ", next_node.depth)


def search_move(s0_node, n_simulation, n_depth):
    for i in range(n_simulation):
        # step 1: select to time step L
        selected_node = s0_node.select(depth=n_depth)
        # step 2: expand an evaluate
        reward = selected_node.evaluate()
        # step 3: backup
        selected_node.update_recursive(reward, s0_node.depth, selected_node.board.turn)


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
    parser.add_argument("--simulations", "-s", help="Simulation numbers. Default: 100", type=int,
                        default=100)  # noqa: E501
    parser.add_argument("--depth", "-d", help="Search depth of simulation. Default: 1", type=int,
                        default=1)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)
    play_games(args.model, args.weights, args.out_directory, args.games, args.pid, args.simulations, args.depth)


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # logging.basicConfig(filename='./out/predict.log', level=logging.INFO)
    run_self_play()
