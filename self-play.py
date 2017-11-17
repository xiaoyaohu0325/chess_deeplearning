from policy import ResnetPolicy
from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter
import logging
import chess

simulations = 100
depth = 3


def play_a_game():
    policy = ResnetPolicy.load_model("./out/model/model_2_128.json")

    for i in range(1):
        # start a new
        start = timer()
        root_node = TreeNode(None, policy=policy)
        next_node = root_node
        moves = 0
        while True:
            start_search = timer()
            search_move(next_node, simulations, depth)
            end_search = timer()
            next_node = next_node.play()
            moves += 1
            print('search move ', end_search - start_search)
            if next_node.board.is_game_over(claim_draw=True):
                break

        next_node.feed_back_winner()

        game_converter.save_pgn_to_hd5(file_path="./out/self_play/test_pgn.h5",
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path="./out/self_play/test_features.h5",
                                       game_tree=root_node)
        end = timer()
        print("game ", i, " finished!  elapsed ", end-start, ", round: ", next_node.depth)


def _print_node_info(node):
    for (action, subnode) in node.children.items():
        from_square_name = chess.square_name(action[0])
        to_square_name = chess.square_name(action[1])
        logging.info("move: %s%s, value: %s", from_square_name, to_square_name, subnode._weights())


def search_move(s0_node, n_simulation, n_depth):
    s0_node.before_search()
    for i in range(n_simulation):
        # step 1: select to time step L
        selected_node = s0_node.select(depth=n_depth)
        # step 2: expand an evaluate
        reward = selected_node.evaluate()
        # step 3: backup
        selected_node.update_recursive(reward, s0_node.depth)
        # if s0_node.depth == 50:
        #     logging.info("iteration %d", i)
        #     _print_node_info(s0_node)


def run_self_play():
    play_a_game()
    # cpus = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool()
    #
    # for i in range(cpus):
    #     pool.apply_async(play_a_game, args=(i,))
    #
    # pool.close()
    # pool.join()


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # logging.basicConfig(filename='./out/predict.log', level=logging.INFO)
    run_self_play()
