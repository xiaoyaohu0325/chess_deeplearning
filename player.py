from game_tree import TreeNode
import chess
from random import randint


class Player(object):
    def __init__(self, name: str, color: bool):
        self.name = name
        self.color = color

    def generate_move(self, fen: str):
        pass


class RandomPlayer(Player):
    def generate_move(self, fen: str):
        board = chess.Board(fen)
        """Randomly choose a legal move to play"""
        if board.is_game_over(claim_draw=True):
            return None
        moves = []
        for legal_move in board.generate_legal_moves():
            moves.append(legal_move)
        if len(moves) > 0:
            move = moves[randint(0, len(moves) - 1)]
            return move


class AIPlayer(Player):
    def __init__(self, name: str, color: bool,
                 policy=None,
                 simulation=40,
                 depth=2):
        Player.__init__(self, name=name, color=color)
        self.policy = policy
        self.n_simulation = simulation
        self.n_depth = depth

    def generate_move(self, fen: str):
        node = TreeNode(parent=None, policy=self.policy, fen=fen)
        if node.board.is_game_over():
            print("Game is over!")
            return None

        self._search_move(node)
        next_node = node.select(depth=1)
        return next_node.move

    def _search_move(self, tree_node):
        for i in range(self.n_simulation):
            # step 1: select to time step L
            selected_node = tree_node.select(depth=self.n_depth)
            # step 2: expand an evaluate
            reward = selected_node.evaluate()
            # step 3: backup
            selected_node.update_recursive(reward)
