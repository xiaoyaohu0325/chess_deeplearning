from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter


def play_game(num_game=1):
    for i in range(num_game):
        # start a new
        start = timer()
        root_node = TreeNode(None)
        next_node = root_node

        while True:
            # start_search = timer()
            next_node = play_a_move(next_node)
            # end_search = timer()
            # print('search move ', end_search - start_search)
            if next_node.board.is_game_over(claim_draw=True):
                break

        next_node.feed_back_winner()
        result = next_node.board.result(claim_draw=True)

        if result != "1/2-1/2":
            game_converter.save_pgn_to_hd5(file_path="./out/self_play/test_pgn.h5",
                                           pgn=next_node.export_pgn_str(),
                                           game_result=result)
            game_converter.features_to_hd5(file_path="./out/self_play/test_features.h5",
                                           game_tree=root_node)
            break

        end = timer()
        print("game", i, "finished!  elapsed", end-start, ", round:", next_node.depth, 'result:', result)


def play_a_move(s0_node):
    # step 1: expand with legal moves
    actions = []
    for move in s0_node.board.generate_legal_moves():
        action = (move.from_square, move.to_square)
        actions.append(action)

    for action in actions:
        sub_node = TreeNode(parent=s0_node,
                            action=action)
        s0_node.children[action] = sub_node
        sub_node.set_prior_prob(1.0/len(actions))

    # step 2: select a move
    (action, selected_node) = max(s0_node.children.items(), key=lambda act_node: act_node[1].get_value())
    # step 3: update pi
    for item in s0_node.children.items():
        (move, node) = item
        s0_node.pi[move[0] * 64 + move[1]] = node.P

    # step 4: play the selected move
    s0_node.children.clear()
    s0_node.children[action] = selected_node
    return selected_node


def run_random_play():
    play_game(100)


if __name__ == '__main__':
    run_random_play()
