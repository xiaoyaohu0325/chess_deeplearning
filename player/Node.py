import numpy as np
from numpy.random import dirichlet
from util.actions import move_to_index
from collections import namedtuple

CounterKey = namedtuple("CounterKey", "board to_play depth idx")
c_PUCT = 3
virtual_loss = 2


class Node:
    """
    The learning rate was set to 0.2 for each game, and was dropped three times (to 0.02, 0.002 and 0.0002 respectively)
    during the course of training. Moves are selected in proportion to the root visit count.
    Dirichlet noise Dir(α) was added to the prior probabilities in the root node; this was scaled in inverse proportion
    to the approximate number of legal moves in a typical position, to a value of α = {0.3, 0.15, 0.03} for chess,
    shogi(日本将棋) and Go respectively. Unless otherwise specified, the training and search algorithm and parameters are
    identical to AlphaGo Zero.
    """
    def __init__(self, parent=None, board=None, move_prob=None, index=0):
        """
        board: chess board
        """
        self.parent = parent
        self.board = board
        self.n = 0 if self.parent is None else self.parent.n+1
        self.index = index
        self.to_play = None
        self.played = False
        self.children = {}
        self.legal_moves = []
        self.move = None
        self.features = None

        self.W = 0
        self.N = 0
        self.Q = 0
        self.P = 0 if move_prob is None else move_prob
        self.U = 0
        self.pi = np.zeros((4672,))
        self.reward = 0

    def __str__(self):
        return "depth: {0:d}, move: {1}, player: {2}".format(self.n,
                                                             "None" if self.move is None else self.move,
                                                             "W" if self.to_play else "B")

    def copy_board_from_parent(self):
        if self.board is not None:
            return
        assert self.parent is not None, "parent must not be None when copying board from parent"
        assert self.parent.board is not None
        self.board = self.parent.board.copy(stack=False)
        self.to_play = self.board.turn

    def counter_key(self) -> namedtuple:
        if self.board is None and self.move is not None:
            self.play_move()
        return CounterKey(self.board.occupied, self.to_play, self.n, self.index)

    def expand_node(self, predict: np.ndarray)->None:
        """Expand leaf node"""
        if self.board is None and self.move is not None:
            self.play_move()

        probs = []
        for move in self.board.generate_legal_moves():
            self.legal_moves.append(move)
            probs.append(predict[move_to_index(move, self.board.turn)])

        if len(self.legal_moves) == 0:
            return False

        noise = dirichlet([0.3] * len(self.legal_moves))    # according to alphazero paper
        for idx, move in enumerate(self.legal_moves):
            probs[idx] += noise[idx]
            self.append_child_node(move, probs[idx])

        return True

    def back_up_value(self, value: float)->None:
        """update: N, W, Q, U"""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

        if not self.is_root():
            self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / (self.N+1)

    def virtual_loss_do(self)->None:
        self.N += virtual_loss
        self.W -= virtual_loss

    def virtual_loss_undo(self)->None:
        self.N -= virtual_loss
        self.W += virtual_loss

    def get_value(self):
        """Calculate and return the value for this node:
        this search control strategy initially prefers actions with high prior probability and
        low visit count, but asympotically prefers actions with high action-value
        """
        if self.Q > 0 or self.U > 0:
            return self.Q + self.U
        else:
            return c_PUCT * self.P * np.sqrt(len(self.parent.children)) / 2

    def select_move_by_score(self)->tuple:
        selected_node = max(self.children.values(), key=lambda act_node: act_node.get_value())
        return selected_node.move

    def select_next_move(self, keep_children=False):
        self.update_pi()

        selected_move = self.select_move_by_score()
        selected_node = self.children[selected_move]

        if not keep_children:
            # option 1: prune all nodes except the selected one
            self.children.clear()
            self.children[selected_move] = selected_node
        else:
            # option 2: keep the direct children nodes and descendant nodes of the selected node
            for sub_node in self.children.values():
                if sub_node != selected_node:
                    sub_node.children.clear()

        return selected_move

    def update_pi(self):
        """At the end of the search selects a move a to play in the root
        position s0, proportional to its exponentiated visit count
        """
        for move, sub_node in self.children.items():
            self.pi[move_to_index(move, self.board.turn)] += sub_node.N

        if self.n > 30:
            self.pi = self._apply_temperature()

        self.pi /= np.sum(self.pi)
        return self.pi

    def _apply_temperature(self):
        beta = 1 / 1e-12
        log_probabilities = np.log(self.pi)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_root(self):
        root_node = self
        while not root_node.is_root():
            root_node = root_node.parent
        return root_node

    def next_node(self):
        """Return the first child of this node"""
        if self.is_leaf():
            return None
        if len(self.children) == 1:
            return list(self.children.values())[0]
        else:
            for sub_node in self.children.values():
                if not sub_node.is_leaf():
                    return sub_node

    def color_to_play(self):
        return "W" if self.to_play else "B"

    def get_msg(self):
        if self.move is not None:
            uci_str = self.move.uci()
        else:
            uci_str = 'None'
        return "%s\nmove: %s, turn: %s\n%s" % \
               (self.get_name(), uci_str, self.color_to_play(), self._weights())

    def get_name(self):
        return "{0:d}-{1:d}".format(self.n, self.index)

    def _weights(self):
        return "W: %f\nQ: %f\nN: %d\nU: %f" % (self.W, self.Q, self.N, self.U)

    def play_move(self, move=None):
        """Play the move of current node, only once"""
        if not self.played:
            self.copy_board_from_parent()
            if move is not None:
                self.move = move
            self.board.push(self.move)
            self.played = True

    def append_child_node(self, move, move_prob=None):
        """Just append child, don't copy board until necessary(expand node or play the move)"""
        sub_node = Node(parent=self, move_prob=move_prob)
        sub_node.move = move
        sub_node.index = len(self.children)
        self.children[move] = sub_node
        return sub_node
