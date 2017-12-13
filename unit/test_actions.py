import unittest
import chess
from util.actions import move_to_index


class TestActions(unittest.TestCase):
    def test_move_north(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("c1c2")
        # move north for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(0, plane)
        self.assertEqual("c1", chess.square_name(square))
        # move south for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(28, plane)
        self.assertEqual("c1", chess.square_name(square))

        m = chess.Move.from_uci("d2d8")
        # move north for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(5, plane)
        self.assertEqual("d2", chess.square_name(square))
        # move south for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(33, plane)
        self.assertEqual("d2", chess.square_name(square))

    def test_move_north_east(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("b1f5")
        # move north east for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(7+(5-1)-1, plane)
        self.assertEqual("b1", chess.square_name(square))
        # move south west for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(35+(5-1)-1, plane)
        self.assertEqual("b1", chess.square_name(square))

    def test_move_east(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("a5c5")
        # move east for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(14+(3-1)-1, plane)
        self.assertEqual("a5", chess.square_name(square))
        # move west for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(42+(3-1)-1, plane)
        self.assertEqual("a5", chess.square_name(square))

    def test_move_south_east(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("a8h1")
        # move south east for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(21+(8-1)-1, plane)
        self.assertEqual("a8", chess.square_name(square))
        # move north west for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(49+(8-1)-1, plane)
        self.assertEqual("a8", chess.square_name(square))

    def test_move_south(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("e6e2")
        # move south for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(28+(6-2)-1, plane)
        self.assertEqual("e6", chess.square_name(square))
        # move north for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(0+(6-2)-1, plane)
        self.assertEqual("e6", chess.square_name(square))

    def test_move_south_west(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("g8e6")
        # move south west for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(35+(8-6)-1, plane)
        self.assertEqual("g8", chess.square_name(square))
        # move north east for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(7+(8-6)-1, plane)
        self.assertEqual("g8", chess.square_name(square))

    def test_move_west(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("f8a8")
        # move west for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(42+(6-1)-1, plane)
        self.assertEqual("f8", chess.square_name(square))
        # move east for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(14+(6-1)-1, plane)
        self.assertEqual("f8", chess.square_name(square))

    def test_move_north_west(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        m = chess.Move.from_uci("d2b4")
        # move north west for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(49+(4-2)-1, plane)
        self.assertEqual("d2", chess.square_name(square))
        # move south east for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(21+(4-2)-1, plane)
        self.assertEqual("d2", chess.square_name(square))

    def test_knight(self):
        """
        56~64
        The next 8 planes encode possible knight moves for that piece.
        1st plane: knight move two squares up and one square right, (rank+2, file+1)
        2nd plane: knight move one square up and two squares right, (rank+1, file+2)
        3rd plane: knight move one square down and two squares right, (rank-1, file+2)
        4th plane: knight move two squares down and one square right, (rank-2, file+1)
        5th plane: knight move two squares down and one square left, (rank-2, file-1)
        6th plane: knight move one square down and two squares left, (rank-1, file-2)
        7th plane: knight move one square up and two squares left, (rank+1, file-2)
        8th plane: knight move two squares up and one square left, (rank+2, file-1)
        :return:
        """
        m = chess.Move.from_uci("d2e4")
        # for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(56, plane)
        self.assertEqual("d2", chess.square_name(square))
        # for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(60, plane)
        self.assertEqual("d2", chess.square_name(square))

        m = chess.Move.from_uci("a4c5")
        # for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(57, plane)
        self.assertEqual("a4", chess.square_name(square))
        # for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(61, plane)
        self.assertEqual("a4", chess.square_name(square))

        m = chess.Move.from_uci("e8g7")
        # for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(58, plane)
        self.assertEqual("e8", chess.square_name(square))
        # for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(62, plane)
        self.assertEqual("e8", chess.square_name(square))

        m = chess.Move.from_uci("c5d3")
        # for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(59, plane)
        self.assertEqual("c5", chess.square_name(square))
        # for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(63, plane)
        self.assertEqual("c5", chess.square_name(square))

    def test_promotion_queue(self):
        """
        0~56
        The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
        in which the piece will be moved, along one of eight relative compass directions
        {N,NE,E,SE,S,SW,W,NW}.
        1st plane: Move north 1 square
        2nd plane: Move north 2 squares
        ...
        56th plane: Move north-west 7 squares
        """
        # promote to queue is just like normal moves
        m = chess.Move(chess.SQUARES[chess.B7], chess.SQUARES[chess.B8], chess.QUEEN)
        # move north for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(0, plane)
        self.assertEqual("b7", chess.square_name(square))
        # move south for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(28, plane)
        self.assertEqual("b7", chess.square_name(square))

        m = chess.Move(chess.SQUARES[chess.C7], chess.SQUARES[chess.B8], chess.QUEEN)
        # move north west for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(49, plane)
        self.assertEqual("c7", chess.square_name(square))
        # move south east for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(21, plane)
        self.assertEqual("c7", chess.square_name(square))

        m = chess.Move(chess.SQUARES[chess.C7], chess.SQUARES[chess.D8], chess.QUEEN)
        # move north east for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(7, plane)
        self.assertEqual("c7", chess.square_name(square))
        # move south west for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(35, plane)
        self.assertEqual("c7", chess.square_name(square))

    def test_under_promotion(self):
        """
        64~73
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
        :return:
        """
        m = chess.Move(chess.SQUARES[chess.B7], chess.SQUARES[chess.B8], chess.ROOK)
        # move north for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(64, plane)
        self.assertEqual("b7", chess.square_name(square))
        # the same plane for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(64, plane)
        self.assertEqual("b7", chess.square_name(square))

        m = chess.Move(chess.SQUARES[chess.C7], chess.SQUARES[chess.B8], chess.BISHOP)
        # move north west for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(68, plane)
        self.assertEqual("c7", chess.square_name(square))
        # move south east for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(71, plane)
        self.assertEqual("c7", chess.square_name(square))

        m = chess.Move(chess.SQUARES[chess.C7], chess.SQUARES[chess.D8], chess.KNIGHT)
        # move north east for white
        idx = move_to_index(m, chess.WHITE)
        plane, square = divmod(idx, 64)
        self.assertEqual(72, plane)
        self.assertEqual("c7", chess.square_name(square))
        # move south west for black
        idx = move_to_index(m, chess.BLACK)
        plane, square = divmod(idx, 64)
        self.assertEqual(69, plane)
        self.assertEqual("c7", chess.square_name(square))
