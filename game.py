import chess
import chess.pgn as pgn
import h5py as h5


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

    def play(self):
        if self.board.is_game_over(claim_draw=True):
            self.game.headers["Result"] = self.board.result(claim_draw=True)
            return None

        fen = self.board.fen()
        move = self.players[self.turn].generate_move(fen)
        self.board.push(move)
        self.game_node = self.game_node.add_variation(move)
        self.turn = not self.turn
        return move

    def fullmove_count(self):
        return self.board.fullmove_number

    def winner_color(self):
        result = self.board.result(claim_draw=True)

        if result == '0-1':
            return chess.BLACK
        elif result == '1-0':
            return chess.WHITE
        elif result == '1/2-1/2':
            return None
        else:
            raise 'game is not over'

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
