import numpy as np
import chess

WHITE = +1
BLACK = -1
EMPTY = 0


class GameState(object):
    """State of a game of Go and some basic functions to interact with it
    """

    def __init__(self, chess_board):
        self.board = np.zeros((8, 8), np.int8)
        self.board.fill(EMPTY)
        # chess.board uses "COLORS = [WHITE, BLACK] = [True, False]"
        if chess_board.turn:
            self.current_player = WHITE
        else:
            self.current_player = BLACK

        piece_map = chess_board.piece_map()

        for square in piece_map:
            piece = piece_map[square]
            # invert the rank
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            if piece.color:
                self.board[rank][file] = WHITE
            else:
                self.board[rank][file] = BLACK
