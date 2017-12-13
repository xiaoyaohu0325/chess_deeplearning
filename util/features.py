import numpy as np
import chess


def extrac_piece_planes(game, board: chess.Board, current_player):
    """
    Extract piece planes for current board state.
    Current player's pieces
    0. pawn
    1. knight
    2. bishop
    3. rook
    4: queue
    5: king
    6-11 are opponent's pieces in the same order
    :param game:
    :param board:
    :param current_player:
    :return: ndarray of shape 8*8*12. The board is oriented to the perspective of the current player.
    """
    assert board is not None, "board must not be None"
    result = np.zeros((8, 8, 14), dtype=np.uint8)

    def _extract_piece(p_type, color, index):
        square_set = board.pieces(p_type, color)
        for square in square_set:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if current_player == chess.BLACK:
                rank = 7 - rank
                file = 7 - file
            result[rank, file, index] = 1

    for idx, piece_type in enumerate(chess.PIECE_TYPES):
        # white
        _extract_piece(piece_type, current_player, idx)
        # black
        _extract_piece(piece_type, not current_player, idx+6)

    repetition_num = game.count_repetitions(board)
    if repetition_num == 1:
        result[:, :, 12] = 1
    elif repetition_num >= 2:
        result[:, :, 13] = 1

    return result


def extract_features(game, node):
    """
    Group 1:

    Feature             planes
    P1 pieces           6
    P2 pieces           6
    Repetitions         2

    Group 2:

    Feature             planes
    Colour              1
    Total move count    1
    P1 castling         2
    P2 castling         2
    No-progress count   1

    Total planes: 119

    The first group of features are repeated for each position in a T = 8-step history.
    The input to the neural network is an N × N × (M*T + L) image stack that represents state using a concatenation of
    T sets of M planes of size N × N .
    Each set of planes represents the board position at a time-step t − T + 1, ..., t,
    and is set to zero for time-steps less than 1.

    The M feature planes are composed of binary feature planes indicating the presence of the player’s pieces,
    with one plane for each piece type, and a second set of planes indicating the presence of the opponent’s pieces.

    Counts are represented by a single real-valued input; other input features are represented
    by a one-hot encoding using the specified number of binary input planes.
    The current player is denoted by P1 and the opponent by P2.


    Training proceeded for 700,000 steps (mini-batches of size 4,096) starting from randomly initialised parameters
    :param game:
    :param node:
    :return:
    """
    def _move_history():
        """ Each set of planes represents the board position at a time-step t − T + 1, ..., t"""
        planes_time_step = 14
        result = np.zeros((8, 8, planes_time_step*8))  # 14*8
        current_node = node
        player = node.board.turn

        for i in range(7, 0, -1):
            if current_node is not None:
                planes = extrac_piece_planes(game, current_node.board, player)
                result[:, :, i*planes_time_step:(i+1)*planes_time_step] = planes
                current_node = current_node.parent

        return result

    def _color_plane(turn):
        if turn == chess.WHITE:
            return np.ones((8, 8), dtype=np.uint8)
        return np.zeros((8, 8), dtype=np.uint8)

    def _total_moves(board: chess.Board):
        return np.full((8, 8), board.fullmove_number, dtype=np.uint16)

    def _castling_planes(board: chess.Board):
        """The current player is denoted by P1 and the opponent by P2"""
        p1 = board.turn
        p2 = not p1
        result = np.zeros((8, 8, 4), dtype=np.uint8)
        if board.has_queenside_castling_rights(p1):
            result[:, :, 0] = 1
        if board.has_kingside_castling_rights(p1):
            result[:, :, 1] = 1
        if board.has_queenside_castling_rights(p2):
            result[:, :, 2] = 1
        if board.has_kingside_castling_rights(p2):
            result[:, :, 3] = 1
        return result

    def _non_process_moves(board: chess.Board):
        return np.full((8, 8), board.halfmove_clock, dtype=np.uint16)

    features = np.empty((8, 8, 119))
    features[:, :, 0:112] = _move_history()
    features[:, :, 112] = _color_plane(node.board.turn)
    features[:, :, 113] = _total_moves(node.board)
    features[:, :, 114:118] = _castling_planes(node.board)
    features[:, :, 118] = _non_process_moves(node.board)
    return features


def bulk_extract_features(game, nodes):
    num_nodes = len(nodes)
    output = np.zeros([num_nodes, 8, 8, 119])
    for i, pos in enumerate(nodes):
        output[i] = extract_features(game, pos)
    return output


def decode_piece(p):
    return ord(p) - ord('A')
