import chess
import chess.pgn as pgn
import h5py as h5
import numpy as np
import collections
from player.Node import Node
from util.features import extract_features
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def board_key(board: chess.Board):
    """Calculate key of board"""
    return (board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
            board.promoted,
            board.turn, board.clean_castling_rights(),
            board.ep_square if board.has_legal_en_passant() else None)


class Game(object):
    def __init__(self, white_player, black_player, generate_features=True):
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
        # Game tree
        self.root_node = Node(board=self.board)
        self.leaf_node = self.root_node
        self.generate_features = generate_features
        if self.generate_features:
            self.leaf_node.features = extract_features(self, self.leaf_node)

    def play(self, move_to_play=None):
        if self._can_claim_draw():
            self.game.headers["Result"] = "1/2-1/2"
            return None, None
        if self.board.is_game_over():
            self.game.headers["Result"] = self.board.result()
            return None, None

        if move_to_play is None:
            node = Node(board=self.board.copy(stack=False))
            move, rate = self.players[self.turn].suggest_move(self, node)
            if self.generate_features:
                self.leaf_node.pi = node.pi     # get pi of the node
        else:
            move = move_to_play
            rate = 0.5

        self.leaf_node = self.leaf_node.append_child_node(move, rate)
        if self.generate_features:
            self.leaf_node.play_move()
            self.leaf_node.features = extract_features(self, self.leaf_node)

        logger.debug("turn {0}, move {1}, full moves {2:d}".format(self.turn, move if move is not None else 'None',
                                                                   self.board.fullmove_number))
        self.board.push(move)
        self.repetitions.update((board_key(self.board),))
        self.game_node = self.game_node.add_variation(move)
        self.turn = self.board.turn
        return move, rate

    def play_to_end(self, max_moves=500):
        """Play the game until game is over"""
        moves = 0
        while True:
            move, win_rate = self.play()
            moves += 1
            if (moves > max_moves) or move is None:
                break
        self.feed_back_rewards()

    def count_repetitions(self, board):
        """Count the repetition number of the board state"""
        return self.repetitions[board_key(board)]

    def _can_claim_draw(self):
        if self.board.can_claim_fifty_moves() or self.repetitions[board_key(self.board)] >= 3:
            return True
        return False

    def winner_color(self):
        result = self.board.result()

        if result == '0-1':
            return chess.BLACK
        elif result == '1-0':
            return chess.WHITE
        else:
            return None

    def feed_back_rewards(self):
        """Invoke this method only when game is over"""
        reward = 0 if self.winner_color() is None else -1
        node = self.leaf_node
        while node is not None:
            node.reward = reward
            node = node.parent
            reward *= -1

    def save_pgn(self, file_path):
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

    def _node_features(self):
        """Convert a game tree into nerual network input
        """
        # iterate moves until game end or leaf node arrived
        current_node = self.root_node
        while current_node is not None:
            features = current_node.features
            r = current_node.reward
            pi = current_node.pi
            yield (features, pi, r)
            current_node = current_node.next_node()

    def save_features(self, file_path):
        h5f = h5.File(file_path)

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            if "features" not in h5f:
                h5f.require_dataset(
                    name='features',
                    dtype=np.uint8,
                    shape=(0, 8, 8, 18),
                    maxshape=(None, 8, 8, 119),
                    chunks=True,
                    compression="lzf")
            if "pi" not in h5f:
                h5f.require_dataset(
                    name='pi',
                    dtype=np.float,
                    shape=(0, 4672),
                    maxshape=(None, 4672),
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
            from_idx = size

            for state, pi, r in self._node_features():
                features.resize((size + 1, 8, 8, 119))
                actions.resize((size + 1, 4672))
                rates.resize((size + 1, 1))
                features[size] = state
                actions[size] = pi
                rates[size] = r
                size += 1
            return from_idx, size
        except Exception as e:
            print("append to hdf5 failed")
            raise e
        finally:
            # processing complete; rename tmp_file to hdf5_file
            h5f.close()
