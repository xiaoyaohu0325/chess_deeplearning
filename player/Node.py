import chess
import numpy as np
from numpy.random import dirichlet
from util.features import action_to_index
from collections import namedtuple

CounterKey = namedtuple("CounterKey", "board to_play depth")
c_PUCT = 3
virtual_loss = 3


class Node:
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
        for move in self.board.generate_legal_moves():
            self.legal_moves.append((move.from_square, move.to_square))

        self.W = 0
        self.N = 0
        self.Q = 0
        self.P = 0 if move_prob is None else move_prob
        self.U = self.P

    def __str__(self):
        return "depth: {0:d}, move: {1}, player: {2}".format(self.n,
                                                             "None" if self.move is None else self.move,
                                                             "W" if self.to_play else "B")

    def counter_key(self) -> namedtuple:
        return CounterKey(tuple(self.board_array()), self.to_play, self.n)

    def expand_node(self, move_probabilities: np.ndarray)->None:
        """Expand leaf node"""
        if len(self.legal_moves) == 0:
            return False

        prob = self.predict_to_prob(move_probabilities)
        for action in self.legal_moves:
            self.play_move(action, move_prob=prob[action_to_index(action)])

        return True

    def is_game_over(self):
        return self.board.is_game_over()

    def back_up_value(self, value: float)->None:
        """update: N, W, Q, U"""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

        if not self.is_root():
            self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)

    def virtual_loss_do(self)->None:
        self.N += virtual_loss
        self.W -= virtual_loss

    def virtual_loss_undo(self)->None:
        self.N -= virtual_loss
        self.W += virtual_loss

    def select_action_by_score(self)->tuple:
        def noise_score(node):
            return node.Q + node.U * (0.75 * node.P + 0.25 * dirichlet([.03, 1])[0]) / (node.P + 1e-8)

        def pure_score(node):
            return node.Q + node.U

        score_func = noise_score if self.n < 30 else pure_score

        selected_node = max(self.children.values(), key=lambda act_node: score_func(act_node))
        return selected_node.move.from_square, selected_node.move.to_square

    def prune_tree(self, prune=True):
        selected_action = self.select_action_by_score()
        selected_node = self.children[selected_action]

        if prune:
            self.children.clear()
            self.children[selected_action] = selected_node

        return selected_action

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

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

    def board_array(self):
        """Return the board state as an array of size 64.
        The lower left index is 0. The value is piece type.
        For example:
        piece type 'P'(white pawn), value = ord('P') - ord('A') = 15
        piece type 'b'(black bishop), value = ord('b') - ord('A') = 33
        """
        result = np.zeros((64,), dtype=np.int8)
        for square, piece in self.board.piece_map().items():
            result[square] = ord(piece.symbol()) - ord('A')
        return result

    def is_legal_move(self, action):
        return action in self.legal_moves

    def predict_to_prob(self, predict):
        """predict is an array of size 128"""
        p_from = predict[:64]
        p_to = predict[64:]
        result = np.zeros((4096,))
        for action in self.legal_moves:
            p_1 = p_from[action[0]]
            p_2 = p_to[action[1]]
            result[action_to_index(action)] = p_1*p_2
        result /= np.sum(result)  # make sure the sum of result is 1
        return result

    def play_move(self, action, move_prob=None):
        if not self.is_legal_move(action):
            return None

        from_square, to_square = action
        board_copy = self.board.copy()

        from_square_name = chess.square_name(from_square)
        to_square_name = chess.square_name(to_square)

        piece = board_copy.piece_type_at(from_square)
        promotion = False
        if piece == chess.PAWN:
            if board_copy.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                promotion = True
            elif board_copy.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                promotion = True

        if promotion:
            # Promote to knight if it can checkmate, otherwise promote to queue
            knight_p = chess.Move.from_uci(from_square_name + to_square_name + 'n')
            board_copy.push(knight_p)
            if board_copy.is_checkmate():
                move = knight_p
            else:
                board_copy.pop()  # remove last move
                move = chess.Move.from_uci(from_square_name + to_square_name + 'q')
                board_copy.push(move)
        else:
            move = chess.Move.from_uci(from_square_name + to_square_name)
            board_copy.push(move)

        sub_node = Node(parent=self, board=board_copy, move_prob=move_prob)
        sub_node.move = move
        sub_node.index = len(self.children)
        self.children[action] = sub_node
        return sub_node

    def result(self):
        """
        Gets the game result.

        ``1-0``, ``0-1`` or ``1/2-1/2`` if the
        game is over. Otherwise the result is undetermined: ``*``.
        """
        return self.board.result()
