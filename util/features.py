import numpy as np
import chess


def extract_features(node):
    return fen_to_features(node.fen())


def bulk_extract_features(nodes):
    num_nodes = len(nodes)
    output = np.zeros([num_nodes, 8, 8, 18], dtype=np.uint8)
    for i, pos in enumerate(nodes):
        output[i] = extract_features(pos)
    return output


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


def action_to_index(action):
    return action[0]*64 + action[1]


def index_to_action(idx):
    return divmod(idx, 64)


def action_to_icu(action):
    return chess.square_name(action[0]) + chess.square_name(action[1])