import chess
import chess.pgn as pgn
import h5py as h5
import collections
from player.Node import Node


def board_key(board: chess.Board):
    """Calculate key of board"""
    return (board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
            board.promoted,
            board.turn, board.clean_castling_rights(),
            board.ep_square if board.has_legal_en_passant() else None)


class Game(object):
    def __init__(self, white_player, black_player):
        self.players = {
            chess.WHITE: white_player,
            chess.BLACK: black_player
        }
        self.turn = chess.WHITE
        self.board = chess.Board()
        self.game = pgn.Game()
        self.game.headers["Event"] = "checkpoint"
        self.game.headers["White"] = white_player.name
        self.game.headers["Black"] = black_player.name
        self.game_node = self.game
        self.is_game_over = False
        self.repetitions = collections.Counter()

    def play(self):
        if self._can_claim_draw():
            self.game.headers["Result"] = "1/2-1/2"
            return None
        if self.board.is_game_over():
            self.game.headers["Result"] = self.board.result()
            return None

        node = Node(board=self.board.copy(stack=False))
        move, rate = self.players[self.turn].suggest_move(node)
        self.board.push(move)
        self.repetitions.update((board_key(self.board),))
        self.game_node = self.game_node.add_variation(move)
        self.turn = not self.turn
        return move, rate

    def count_repetitions(self, board):
        """Count the repetition number of the board state"""
        return self.repetitions[board_key(board)]

    def _can_claim_draw(self):
        if self.board.can_claim_fifty_moves() or self.repetitions[board_key(self.board)] >= 3:
            return True
        return False

    def winner_color(self):
        result = self.board.result(claim_draw=True)

        if result == '0-1':
            return chess.BLACK
        elif result == '1-0':
            return chess.WHITE
        else:
            return None

    def save_to_h5(self, file_path):
        exporter = pgn.StringExporter(headers=True)
        pgn_str = self.game.accept(exporter)

        h5f = h5.File(file_path)
        white_name = self.players[chess.WHITE].name
        black_name = self.players[chess.BLACK].name

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            dt = h5.special_dtype(vlen=bytes)
            if "draw" not in h5f:
                h5f.require_dataset(
                    name='draw',
                    dtype=dt,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=True,
                    compression="lzf")
            if white_name not in h5f:
                h5f.require_dataset(
                    name=white_name,
                    dtype=dt,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=True,
                    compression="lzf")
            if black_name not in h5f:
                h5f.require_dataset(
                    name=black_name,
                    dtype=dt,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=True,
                    compression="lzf")

            winner_color = self.winner_color()
            if winner_color == chess.WHITE:
                dest = h5f[white_name]
            elif winner_color == chess.BLACK:
                dest = h5f[black_name]
            else:
                dest = h5f["draw"]

            size = len(dest)
            dest.resize((size + 1,))
            dest[size] = pgn_str
        except Exception as e:
            print("append to hdf5 failed")
            raise e
        finally:
            h5f.close()
