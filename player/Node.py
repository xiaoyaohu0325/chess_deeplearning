import simple_chess as chess
import chess.pgn as pgn
import numpy as np
from numpy.random import dirichlet
from util.features import move_to_index
from collections import namedtuple

CounterKey = namedtuple("CounterKey", "board to_play depth")
c_PUCT = 3
virtual_loss = 3


class Node:
    """
    A move in chess may be described in two parts: selecting the piece to move, and then selecting
    among the legal moves for that piece. We represent the policy π(a|s) by a 8 × 8 × 73 stack of
    planes encoding a probability distribution over 4,672 possible moves. Each of the 8 × 8 positions
    identifies the square from which to “pick up” a piece.

    The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
    in which the piece will be moved, along one of eight relative compass directions
    {N,NE,E,SE,S,SW,W,NW}.
    1st plane: Move north 1 square
    2nd plane: Move north 2 squares
    ...
    56th plane: Move north-west 7 squares

    The next 8 planes encode possible knight moves for that piece.
    1st plane: knight move two squares up and one square right, (rank+2, file+1)
    2nd plane: knight move one square up and two squares right, (rank+1, file+2)
    3rd plane: knight move one square down and two squares right, (rank-1, file+2)
    4th plane: knight move two squares down and one square right, (rank-2, file+1)
    5th plane: knight move two squares down and one square left, (rank-2, file-1)
    6th plane: knight move one square down and two squares left, (rank-1, file-2)
    7th plane: knight move one square up and two squares left, (rank+1, file-2)
    8th plane: knight move two squares up and one square left, (rank+2, file-1)

    The final 9 planes encode possible underpromotions for pawn moves or captures in two possible diagonals,
    to knight, bishop or rook respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.
    1st plane: move forward, promote to rook
    2nd plane: move forward, promote to bishop
    3rd plane: move forward, promote to knight
    4th plane: capture up left, promote to rook
    5th plane: capture up left, promote to bishop
    6th plane: capture up left, promote to knight
    7th plane: capture up right, promote to rook
    8th plane: capture up right, promote to bishop
    9th plane: capture up right, promote to knight

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
        self.board = board if board is not None else chess.Board()
        self.n = 0 if self.parent is None else self.parent.n+1
        self.index = index
        self.to_play = self.board.turn
        self.children = {}
        self.legal_moves = []
        self.move = None

        self.W = 0
        self.N = 0
        self.Q = 0
        self.P = 0 if move_prob is None else move_prob
        self.U = 0
        self.pi = np.zeros((128,))
        self.reward = 0

    def __str__(self):
        return "depth: {0:d}, move: {1}, player: {2}".format(self.n,
                                                             "None" if self.move is None else self.move,
                                                             "W" if self.to_play else "B")

    def counter_key(self) -> namedtuple:
        return CounterKey(self.board.occupied, self.to_play, self.n)

    def expand_node(self, predict: np.ndarray)->None:
        """Expand leaf node"""
        """predict is an array of size 128"""
        p_from = predict[:64]
        p_to = predict[64:]
        result = np.zeros((4096,))

        for move in self.board.generate_legal_moves():
            self.legal_moves.append(move)
            p_1 = p_from[move.from_square]
            p_2 = p_to[move.to_square]
            result[move_to_index(move)] = p_1 * p_2

        if len(self.legal_moves) == 0:
            return False

        # add noise
        # if self.n < 30:
        #     noise = dirichlet([.03] * len(self.legal_moves))
        #     for idx, move in enumerate(self.legal_moves):
        #         result[move_to_index(move)] += noise[idx]

        result /= np.sum(result)  # make sure the sum of result is 1
        for move in self.legal_moves:
            self.play_move(move, move_prob=result[move_to_index(move)])

        return True

    def is_game_over(self):
        return self.board.is_game_over()

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
            self.pi[move.from_square] += sub_node.N
            self.pi[64 + move.to_square] += sub_node.N

        self.pi /= np.sum(self.pi)
        return self.pi

    def _feedback_reward(self, value):
        self.reward = value

        node = self.parent
        while node is not None:
            value = value * -1
            node.reward = value
            node = node.parent

    def feed_back_winner(self, force=False):
        """When game is over, it is then scored to give a final reward of r_T {-1,0,+1}
        The data for each time-step t is stored as (s_t,Pi_t,z_t) where z_t = ±r_T is the
        game winner from the perspective of the current player at step t
        """
        value = 0
        if not force:
            if not self.board.is_game_over():
                raise ValueError("this method can be invoked only when game is over")

            result = self.result()
            if result == "0-1":
                value = -1
            elif result == "1-0":
                value = -1
            elif result == "1/2-1/2":
                value = 0

        self._feedback_reward(value)

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

    def fen(self):
        return self.board.fen()

    def is_legal_move(self, move):
        return move in self.legal_moves

    def play_move(self, move, move_prob=None):
        board_copy = self.board.copy(stack=False)
        board_copy.push(move)

        sub_node = Node(parent=self, board=board_copy, move_prob=move_prob)
        sub_node.move = move
        sub_node.index = len(self.children)
        self.children[move] = sub_node
        return sub_node

    def result(self):
        """
        Gets the game result.

        ``1-0``, ``0-1`` or ``1/2-1/2`` if the
        game is over. Otherwise the result is undetermined: ``*``.
        """
        result = self.board.result()
        if result == '*':
            return '1/2-1/2'
        return result
