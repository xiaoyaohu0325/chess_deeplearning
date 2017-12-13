import chess


def move_to_index(move: chess.Move, current_player):
    """
    A move in chess may be described in two parts: selecting the piece to move, and then selecting
    among the legal moves for that piece. We represent the policy π(a|s) by a 8 × 8 × 73 stack of
    planes encoding a probability distribution over 4,672 possible moves. Each of the 8 × 8 positions
    identifies the square from which to “pick up” a piece.

    0~56
    The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
    in which the piece will be moved, along one of eight relative compass directions
    {N,NE,E,SE,S,SW,W,NW}.
    1st plane: Move north 1 square
    2nd plane: Move north 2 squares
    ...
    56th plane: Move north-west 7 squares

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
    :param move:
    :param current_player:
    :return:
    """
    rank_dist = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    file_dist = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    if current_player == chess.BLACK:
        # From black orientation, flip up down
        rank_dist *= -1
        file_dist *= -1

    if move.promotion:
        if file_dist == 0:
            if move.promotion == chess.ROOK:
                plane_index = 64
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.BISHOP:
                plane_index = 65
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.KNIGHT:
                plane_index = 66
                return plane_index * 64 + move.from_square
        if file_dist == -1:
            if move.promotion == chess.ROOK:
                plane_index = 67
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.BISHOP:
                plane_index = 68
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.KNIGHT:
                plane_index = 69
                return plane_index * 64 + move.from_square
        if file_dist == 1:
            if move.promotion == chess.ROOK:
                plane_index = 70
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.BISHOP:
                plane_index = 71
                return plane_index * 64 + move.from_square
            elif move.promotion == chess.KNIGHT:
                plane_index = 72
                return plane_index * 64 + move.from_square

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
    """
    if rank_dist == 2 and file_dist == 1:
        plane_index = 56
        return plane_index * 64 + move.from_square
    elif rank_dist == 1 and file_dist == 2:
        plane_index = 57
        return plane_index * 64 + move.from_square
    elif rank_dist == -1 and file_dist == 2:
        plane_index = 58
        return plane_index * 64 + move.from_square
    elif rank_dist == -2 and file_dist == 1:
        plane_index = 59
        return plane_index * 64 + move.from_square
    elif rank_dist == -2 and file_dist == -1:
        plane_index = 60
        return plane_index * 64 + move.from_square
    elif rank_dist == -1 and file_dist == -2:
        plane_index = 61
        return plane_index * 64 + move.from_square
    elif rank_dist == 1 and file_dist == -2:
        plane_index = 62
        return plane_index * 64 + move.from_square
    elif rank_dist == 2 and file_dist == -1:
        plane_index = 63
        return plane_index * 64 + move.from_square

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
    if file_dist == 0 and rank_dist > 0:
        # move north, from plane [0,7)
        plane_index = rank_dist - 1
        return plane_index * 64 + move.from_square
    elif file_dist > 0 and rank_dist > 0:
        # move north east, from plane [7,14)
        assert file_dist == rank_dist, "Invalid move {0}".format(move.uci())
        plane_index = (rank_dist - 1) + 7
        return plane_index * 64 + move.from_square
    elif rank_dist == 0 and file_dist > 0:
        # move east, from plane [14,21)
        plane_index = (file_dist - 1) + 14
        return plane_index * 64 + move.from_square
    elif rank_dist < 0 < file_dist:
        # move south east, from plane [21,28)
        assert file_dist == -rank_dist, "Invalid move {0}".format(move.uci())
        plane_index = (file_dist - 1) + 21
        return plane_index * 64 + move.from_square
    elif file_dist == 0 and rank_dist < 0:
        # move south, from plane [28,35)
        plane_index = (-rank_dist - 1) + 28
        return plane_index * 64 + move.from_square
    elif file_dist < 0 and rank_dist < 0:
        # move south west, from plane [35,42)
        assert file_dist == rank_dist, "Invalid move {0}".format(move.uci())
        plane_index = (-rank_dist - 1) + 35
        return plane_index * 64 + move.from_square
    elif rank_dist == 0 and file_dist < 0:
        # move west, from plane [42,49)
        plane_index = (-file_dist - 1) + 42
        return plane_index * 64 + move.from_square
    elif file_dist < 0 < rank_dist:
        # move north west, from plane [49,56)
        assert -file_dist == rank_dist, "Invalid move {0}".format(move.uci())
        plane_index = (rank_dist - 1) + 49
        return plane_index * 64 + move.from_square

    raise ValueError("Move {0} can not be converted to action plane".format(move.uci()))

#
# def index_to_move(index: int, board: chess.Board, current_player):
#     plane_index, from_square = divmod(index, 64)
#     """
#     0~56
#     The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7]
#     in which the piece will be moved, along one of eight relative compass directions
#     {N,NE,E,SE,S,SW,W,NW}.
#     1st plane: Move north 1 square
#     2nd plane: Move north 2 squares
#     ...
#     56th plane: Move north-west 7 squares
#     """
#     if -1 < plane_index < 7:
#         # move north
