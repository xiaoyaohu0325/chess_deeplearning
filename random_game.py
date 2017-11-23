from game_tree import TreeNode
from timeit import default_timer as timer
from preprocessing import game_converter
import argparse
import os


def play_game(num_game, pid, outdir):
    games = 0
    while True:
        if games >= num_game:
            break
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

        games += 1
        end = timer()
        print("game", games, "finished!  elapsed", end - start, ", round:", next_node.depth, 'result:', result)
        game_converter.save_pgn_to_hd5(file_path=os.path.join(outdir, "pgn_{0:d}.h5".format(pid)),
                                       pgn=next_node.export_pgn_str(),
                                       game_result=result)
        game_converter.features_to_hd5(file_path=os.path.join(outdir, "features_{0:d}.h5".format(pid)),
                                       game_tree=root_node)


def play_a_move(s0_node):
    # step 1: expand with legal moves
    actions = []
    for move in s0_node.board.generate_legal_moves():
        action = (move.from_square, move.to_square)
        actions.append(action)

    s0_node.N = len(actions)

    for i in range(len(actions)):
        action = actions[i]
        sub_node = TreeNode(parent=s0_node,
                            action=action,
                            index=i)
        s0_node.children[action] = sub_node
        sub_node.set_prior_prob(1.0/len(actions))
        sub_node.N = 1

    # step 3: update pi
    s0_node.update_pi()

    # step 2: select a move
    (action, selected_node) = max(s0_node.children.items(), key=lambda act_node: act_node[1].get_value())

    # step 4: play the selected move
    s0_node.children.clear()
    s0_node.children[action] = selected_node
    return selected_node


def run_random_play(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform random self play to generate data.')
    # required args
    parser.add_argument("games", help="Number of games to generate.")
    parser.add_argument("pid", help="unique id of the generated h5 file.")
    parser.add_argument("outdir", help="directory where game data will be saved.")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    play_game(int(args.games), args.pid, args.outdir)


if __name__ == '__main__':
    run_random_play()
