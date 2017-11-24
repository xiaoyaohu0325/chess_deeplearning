import unittest
from .game_converter import fen_pieces_to_board, analyze_legal_moves, get_piece_index
import chess


class TestConverter(unittest.TestCase):
    def test_fen_to_board(self):
        """rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"""
        board = fen_pieces_to_board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
        # rank 8: rnbqkbnr
        self.assertEqual(board[7][0], decode_piece('r'))
        self.assertEqual(board[7][1], decode_piece('n'))
        self.assertEqual(board[7][2], decode_piece('b'))
        self.assertEqual(board[7][3], decode_piece('q'))
        self.assertEqual(board[7][4], decode_piece('k'))
        self.assertEqual(board[7][5], decode_piece('b'))
        self.assertEqual(board[7][6], decode_piece('n'))
        self.assertEqual(board[7][7], decode_piece('r'))
        # rank 7: pppppppp
        self.assertEqual(board[6][0], decode_piece('p'))
        self.assertEqual(board[6][1], decode_piece('p'))
        self.assertEqual(board[6][2], decode_piece('p'))
        self.assertEqual(board[6][3], decode_piece('p'))
        self.assertEqual(board[6][4], decode_piece('p'))
        self.assertEqual(board[6][5], decode_piece('p'))
        self.assertEqual(board[6][6], decode_piece('p'))
        self.assertEqual(board[6][7], decode_piece('p'))
        # rank 6: 8
        self.assertEqual(board[5][0], 0)
        self.assertEqual(board[5][1], 0)
        self.assertEqual(board[5][2], 0)
        self.assertEqual(board[5][3], 0)
        self.assertEqual(board[5][4], 0)
        self.assertEqual(board[5][5], 0)
        self.assertEqual(board[5][6], 0)
        self.assertEqual(board[5][7], 0)
        # rank 5: 8
        self.assertEqual(board[4][0], 0)
        self.assertEqual(board[4][1], 0)
        self.assertEqual(board[4][2], 0)
        self.assertEqual(board[4][3], 0)
        self.assertEqual(board[4][4], 0)
        self.assertEqual(board[4][5], 0)
        self.assertEqual(board[4][6], 0)
        self.assertEqual(board[4][7], 0)
        # rank 4: 4P3
        self.assertEqual(board[3][0], 0)
        self.assertEqual(board[3][1], 0)
        self.assertEqual(board[3][2], 0)
        self.assertEqual(board[3][3], 0)
        self.assertEqual(board[3][4], decode_piece('P'))
        self.assertEqual(board[3][5], 0)
        self.assertEqual(board[3][6], 0)
        self.assertEqual(board[3][7], 0)
        # rank 3: 8
        self.assertEqual(board[2][0], 0)
        self.assertEqual(board[2][1], 0)
        self.assertEqual(board[2][2], 0)
        self.assertEqual(board[2][3], 0)
        self.assertEqual(board[2][4], 0)
        self.assertEqual(board[2][5], 0)
        self.assertEqual(board[2][6], 0)
        self.assertEqual(board[2][7], 0)
        # rank 2: PPPP1PPP
        self.assertEqual(board[1][0], decode_piece('P'))
        self.assertEqual(board[1][1], decode_piece('P'))
        self.assertEqual(board[1][2], decode_piece('P'))
        self.assertEqual(board[1][3], decode_piece('P'))
        self.assertEqual(board[1][4], 0)
        self.assertEqual(board[1][5], decode_piece('P'))
        self.assertEqual(board[1][6], decode_piece('P'))
        self.assertEqual(board[1][7], decode_piece('P'))
        # rank 1: RNBQKBNR
        self.assertEqual(board[0][0], decode_piece('R'))
        self.assertEqual(board[0][1], decode_piece('N'))
        self.assertEqual(board[0][2], decode_piece('B'))
        self.assertEqual(board[0][3], decode_piece('Q'))
        self.assertEqual(board[0][4], decode_piece('K'))
        self.assertEqual(board[0][5], decode_piece('B'))
        self.assertEqual(board[0][6], decode_piece('N'))
        self.assertEqual(board[0][7], decode_piece('R'))

    def test_legal_moves(self):
        board = chess.Board()  # initial board
        result = analyze_legal_moves(board)

        self.assertEqual(len(result), 10)   # ten pieces can move at start

        n_moves = result.get(1)  # b1, whose index is 1

        self.assertEqual(len(n_moves), 2)
        self.assertTrue((1, chess.square(file_index=0, rank_index=2)) in n_moves)    # b1 knight can move to a2
        self.assertTrue((1, chess.square(file_index=2, rank_index=2)) in n_moves)    # b1 knight can move to c2

        for i in range(8):
            p_moves = result.get(8 + i)
            # pawn can move forward 1 or 2 squares
            self.assertTrue((chess.square(file_index=i, rank_index=1), chess.square(file_index=i, rank_index=2)) in p_moves)
            self.assertTrue((chess.square(file_index=i, rank_index=1), chess.square(file_index=i, rank_index=3)) in p_moves)

    def test_piece_index(self):
        board = chess.Board()  # initial board
        for i in range(16):
            self.assertEqual(get_piece_index(board, i), i)

        board.set_fen("1n2kb2/4p3/b1p3rp/5PP1/pP1p2P1/N2n3P/PB1P2K1/1r2QBNR w - - 22 51")

        self.assertEqual(get_piece_index(board, 4), 0)  # Q
        self.assertEqual(get_piece_index(board, 5), 1)  # B
        self.assertEqual(get_piece_index(board, 6), 2)  # N
        self.assertEqual(get_piece_index(board, 7), 3)  # R
        self.assertEqual(get_piece_index(board, 37), 12)  # P
        self.assertEqual(get_piece_index(board, 38), 13)  # P


def decode_piece(p):
    return ord(p) - ord('A')
