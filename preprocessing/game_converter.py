import numpy as np
import chess
import h5py as h5
from util.features import extract_features


def fen_to_features(fen: str):
    """A FEN record contains six fields. The separator between fields is a space. The fields are:
    1. Piece placement (from white's perspective). Each rank is described, starting
    with rank 8 and ending with rank 1; within each rank, the contents of each square
    are described from file "a" through file "h". Following the Standard Algebraic
    Notation (SAN), each piece is identified by a single letter taken from the standard
    English names (pawn = "P", knight = "N", bishop = "B", rook = "R", queen = "Q" and
    king = "K").[1] White pieces are designated using upper-case letters ("PNBRQK")
    while black pieces use lowercase ("pnbrqk"). Empty squares are noted using digits 1
    through 8 (the number of empty squares), and "/" separates ranks.
    2. Active colour. "w" means White moves next, "b" means Black.
    3. Castling availability. If neither side can castle, this is "-". Otherwise, this
    has one or more letters: "K" (White can castle kingside), "Q" (White can castle
    queenside), "k" (Black can castle kingside), and/or "q" (Black can castle queenside).
    4. En passant target square in algebraic notation. If there's no en passant target
    square, this is "-". If a pawn has just made a two-square move, this is the position
    "behind" the pawn. This is recorded regardless of whether there is a pawn in position
    to make an en passant capture.[2]
    5. Halfmove clock: This is the number of halfmoves since the last capture or pawn
    advance. This is used to determine if a draw can be claimed under the fifty-move rule.
    6. Fullmove number: The number of the full move. It starts at 1, and is incremented
    after Black's move.

    An example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
    1. K position of white player
    2. Q position of white player
    3. R position of white player
    4. N position of white player
    5. B position of white player
    6. P position of white player
    7. k position of black player
    8. q position of black player
    9. r position of black player
    10. n position of black player
    11. b position of black player
    12. p position of black player
    13. White K side castiling is available
    14. White Q side castiling is available
    15. Black k side castiling is available
    16. Black q side castiling is available
    17. En passant target square
    18. white player set to 1, black player set to 0
    """
    features = np.zeros((8, 8, 18), dtype=np.uint8)

    tokens = fen.split(" ")
    board = fen_pieces_to_board(tokens[0])
    player = tokens[1]
    castling = tokens[2]
    en_passant = tokens[3]

    features[:, :, 0] = board == decode_piece('K')
    features[:, :, 1] = board == decode_piece('Q')
    features[:, :, 2] = board == decode_piece('R')
    features[:, :, 3] = board == decode_piece('N')
    features[:, :, 4] = board == decode_piece('B')
    features[:, :, 5] = board == decode_piece('P')
    features[:, :, 6] = board == decode_piece('k')
    features[:, :, 7] = board == decode_piece('q')
    features[:, :, 8] = board == decode_piece('r')
    features[:, :, 9] = board == decode_piece('n')
    features[:, :, 10] = board == decode_piece('b')
    features[:, :, 11] = board == decode_piece('p')

    # default castling='-'
    features[:, :, 12] = 0
    features[:, :, 13] = 0
    features[:, :, 14] = 0
    features[:, :, 15] = 0

    if castling.find('K') >= 0:
        features[:, :, 12] = 1
    if castling.find('Q') >= 0:
        features[:, :, 13] = 1
    if castling.find('k') >= 0:
        features[:, :, 14] = 1
    if castling.find('q') >= 0:
        features[:, :, 15] = 1

    features[:, :, 16] = 0
    if en_passant != "-":
        file = en_passant[0]
        rank = en_passant[1]
        features[8-int(rank), ord(file) - ord('a'), 16] = 1

    features[:, :, 17] = player == 'w'
    return features


def decode_piece(p):
    return ord(p) - ord('A')


def fen_pieces_to_board(pieces: str):
    """lower left as origin, the same as the chess library"""
    board = np.zeros((8, 8), dtype=np.uint8)
    ranks = pieces.split("/")
    for i in range(0, 8):
        rank = ranks[8 - (i + 1)]
        j = 0
        for piece in rank:
            if piece.isnumeric():
                blank_count = int(piece)
                for k in range(blank_count):
                    j += 1
            else:
                board[i][j] = ord(piece) - ord('A')
                j += 1

    return board


def save_pgn_to_hd5(file_path, pgn, game_result):
    h5f = h5.File(file_path)

    try:
        # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
        dt = h5.special_dtype(vlen=bytes)
        if "draw" not in h5f:
            h5f.require_dataset(
                name='draw',
                dtype=dt,
                shape=(0, ),
                maxshape=(None, ),
                chunks=True,
                compression="lzf")
        if "white" not in h5f:
            h5f.require_dataset(
                name='white',
                dtype=dt,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                compression="lzf")
        if "black" not in h5f:
            h5f.require_dataset(
                name='black',
                dtype=dt,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                compression="lzf")

        if game_result == "1-0":
            dest = h5f["white"]
        elif game_result == "0-1":
            dest = h5f["black"]
        else:
            dest = h5f["draw"]

        size = len(dest)
        dest.resize((size + 1,))
        dest[size] = pgn
    except Exception as e:
        print("append to hdf5 failed")
        raise e
    finally:
        # processing complete; rename tmp_file to hdf5_file
        h5f.close()


def convert_game(game_tree):
    """Convert a game tree into nerual network input
    """
    root_node = game_tree.get_root()
    if root_node.is_leaf():
        raise ValueError("no moves available")

    # iterate moves until game end or leaf node arrived
    current_node = root_node
    while True:
        features = extract_features(current_node)
        r = current_node.reward
        pi = current_node.pi
        yield (features, pi, r)
        current_node = current_node.next_node()
        if current_node is None or current_node.is_leaf():
            break


def features_to_hd5(file_path, game_tree):
    h5f = h5.File(file_path)

    try:
        # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
        if "features" not in h5f:
            h5f.require_dataset(
                name='features',
                dtype=np.uint8,
                shape=(0, 8, 8, 18),
                maxshape=(None, 8, 8, 18),
                chunks=True,
                compression="lzf")
        if "pi" not in h5f:
            h5f.require_dataset(
                name='pi',
                dtype=np.float,
                shape=(0, 128),
                maxshape=(None, 128),
                chunks=True,
                compression="lzf")
        if "rewards" not in h5f:
            h5f.require_dataset(
                name='rewards',
                dtype=np.int8,
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression="lzf")

        features = h5f["features"]
        actions = h5f["pi"]
        rates = h5f["rewards"]
        size = len(features)

        for state, pi, r in convert_game(game_tree):
            features.resize((size + 1, 8, 8, 18))
            actions.resize((size + 1, 128))
            rates.resize((size + 1, 1))
            features[size] = state
            actions[size] = pi
            rates[size] = r
            size += 1

    except Exception as e:
        print("append to hdf5 failed")
        raise e
    finally:
        # processing complete; rename tmp_file to hdf5_file
        h5f.close()


def policy_index_to_action(index):
    p_index = int(index/64)
    to_square = index % 64
    return p_index, to_square


def get_piece_index(board, square):
    """
    Get the piece index of the specified square
    :param board:
    :param square:
    :return:
    """
    piece_squares = [key for key, value in board.piece_map().items() if value.color == board.turn]
    piece_squares = sorted(piece_squares)
    return piece_squares.index(square)


def analyze_legal_moves(board: chess.Board):
    """Analyze legal move of current board state.
    return a dict whose keys is index of movable piece(0-15).
    and value is an array of moves.
    """
    result = dict()
    piece_squares = [key for key, value in board.piece_map().items() if value.color == board.turn]
    piece_squares = sorted(piece_squares)

    for move in board.generate_legal_moves():
        idx = piece_squares.index(move.from_square)

        if result.get(idx) is None:
            result[idx] = []

        result[idx].append((move.from_square, move.to_square))

    return result


def check_pi(pi):
    """
    pi is of shape (4096,)
    :param pi:
    :return:
    """
    total = 0
    for i in range(len(pi)):
        if pi[i] > 0:
            action = policy_index_to_action(i)
            to_square = chess.square_name(action[1])
            print('piece order: {0}, to: {1}'.format(action[0], to_square), ", probability:", pi[i])
            total += pi[i]
    print('total probability:', total)
