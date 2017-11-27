import chess
import numpy as np
from util.features import action_to_index


class Position:
    def __init__(self, board=None, move_prob=None):
        """
        board: chess board
        """
        self.board = board if board is not None else chess.Board()
        self.n = 0
        self.move_prob = 0 if move_prob is None else move_prob
        self.to_play = self.board.turn
        self.legal_moves = []
        self.move = None
        for move in self.board.generate_legal_moves():
            self.legal_moves.append((move.from_square, move.to_square))

    def __str__(self):
        return "depth: {0:d}, move: {1}, player: {2}".format(self.n,
                                                             "None" if self.move is None else self.move,
                                                             "W" if self.to_play else "B")

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

    def play_move(self, action, mutate=False, move_prob=None):
        if not self.is_legal_move(action):
            return None

        from_square, to_square = action

        pos = self if mutate else Position(self.board.copy(), move_prob=move_prob)

        from_square_name = chess.square_name(from_square)
        to_square_name = chess.square_name(to_square)

        piece = pos.board.piece_type_at(from_square)
        promotion = False
        if piece == chess.PAWN:
            if pos.board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                promotion = True
            elif pos.board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                promotion = True

        if promotion:
            # 如果升变成马能一步杀就升变成马，否则升变成后
            knight_p = chess.Move.from_uci(from_square_name + to_square_name + 'n')
            pos.board.push(knight_p)
            if pos.board.is_checkmate():
                pos.move = knight_p
            else:
                pos.board.pop()  # remove last move
                pos.move = chess.Move.from_uci(from_square_name + to_square_name + 'q')
                pos.board.push(pos.move)
        else:
            pos.move = chess.Move.from_uci(from_square_name + to_square_name)
            pos.board.push(pos.move)

        # move number increments
        pos.n = self.n + 1
        return pos

    def result(self):
        """
        Gets the game result.

        ``1-0``, ``0-1`` or ``1/2-1/2`` if the
        game is over. Otherwise the result is undetermined: ``*``.
        """
        return self.board.result()
