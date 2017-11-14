from policy import ResnetPolicy
from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter

simulations = 100
depth = 3
max_moves = 300


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
            if moves > max_moves or next_node.board.is_game_over(claim_draw=True):
                break

        if moves > max_moves:
            next_node.feed_back_winner(force=True)
        else:
            next_node.feed_back_winner()

        game_converter.save_pgn_to_hd5(file_path="./out/self_play/test_pgn.h5",
                                       pgn=next_node.export_pgn_str(),
                                       game_result=next_node.board.result(claim_draw=True))
        game_converter.features_to_hd5(file_path="./out/self_play/test_features.h5",
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
        selected_node.update_recursive(reward)


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
    run_self_play()
