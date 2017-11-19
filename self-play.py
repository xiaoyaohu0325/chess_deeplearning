from policy import ResnetPolicy
from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter
import logging
import chess
import argparse


def play_games(games, pid, simulations, depth):
    policy = ResnetPolicy.load_model("./out/model/model_2_128.json")

    for i in range(games):
        # start a new
        start = timer()
        root_node = TreeNode(None, policy=policy)
        next_node = root_node
        moves = 0
        while True:
            # start_search = timer()
            search_move(next_node, simulations, depth)
            # end_search = timer()
            next_node = next_node.play()
            moves += 1
            # print('search move ', end_search - start_search)
            if next_node.board.is_game_over(claim_draw=True):
                break

        next_node.feed_back_winner()

        game_converter.save_pgn_to_hd5(file_path="./out/self_play/test_pgn_" + str(pid) + ".h5",
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path="./out/self_play/test_features_" + str(pid) + ".h5",
                                       game_tree=root_node)
        end = timer()
        print("game ", i, " finished!  elapsed ", end-start, ", round: ", next_node.depth)


def _print_node_info(node):
    for (action, subnode) in node.children.items():
        from_square_name = chess.square_name(action[0])
        to_square_name = chess.square_name(action[1])
        logging.info("move: %s%s, value: %s", from_square_name, to_square_name, subnode._weights())


def search_move(s0_node, n_simulation, n_depth):
    for i in range(n_simulation):
        # step 1: select to time step L
        selected_node = s0_node.select(depth=n_depth)
        # step 2: expand an evaluate
        reward = selected_node.evaluate()
        # step 3: backup
        selected_node.update_recursive(reward, s0_node.depth, selected_node.board.turn)
        # if s0_node.depth == 50:
        #     logging.info("iteration %d", i)
        #     _print_node_info(s0_node)


def run_self_play(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform self play to generate data.')
    # required args
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
    play_games(args.games, args.pid, args.simulations, args.depth)


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # logging.basicConfig(filename='./out/predict.log', level=logging.INFO)
    run_self_play()
