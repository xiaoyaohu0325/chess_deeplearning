from policy import ResnetPolicy
from game_tree import TreeNode
# import multiprocessing
from timeit import default_timer as timer
from preprocessing import game_converter
import os

simulations = 20
depth = 1


def play_a_game(pid):
    policy = ResnetPolicy.load_model("./out/model.json")

    for i in range(1):
        # start a new
        start = timer()
        root_node = TreeNode(None, policy=policy)
        next_node = root_node
        while True:
            start_search = timer()
            search_move(next_node, simulations, depth)
            end_search = timer()
            print("search move elapsed: ", end_search - start_search)
            next_node = next_node.play()

            if next_node.board.is_game_over(claim_draw=True):
                break

        next_node.feed_back_winner()

        game_converter.save_pgn_to_hd5(file_path="./out/self_play/self-play-ai.hdf",
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path="./out/self_play/features-ai.hdf",
                                       game_tree=root_node)
        end = timer()
        print("pid ", pid, ", game ", i, " finished!  elapsed ", end-start)


def search_move(s0_node, n_simulation, n_depth):
    for i in range(n_simulation):
        # step 1: select to time step L
        selected_node = s0_node.select(depth=n_depth)
        # step 2: expand an evaluate
        reward = selected_node.evaluate()
        # step 3: backup
        selected_node.update_recursive(reward)


def run_self_play():
    play_a_game(1)
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
    run_self_play()
