import numpy as np
import chess
import chess.pgn as pgn
from preprocessing import game_converter
import logging
# from policy import ResnetPolicy

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
C_PUCT = 3


class TreeNode(object):
    """Each node s in the search tree contains edges (s, a) for all legal actions a belongs to A(s).
    Each edge stores a set of statistics, {N(s,a),W(s,a),Q(s,a),P(s,a)},
    N (s, a) is the visit count, W (s, a) is the total action-value, Q(s, a) is the mean action-value,
    and P(s, a) is the prior probability of selecting that edge.
    """

    def __init__(self, parent,
                 policy=None,
                 fen=None,
                 action=None   # (from_square_index, to_square_index)
                 ):
        self.parent = parent
        if fen is not None:
            self.fen = fen
        elif parent is not None:
            self.fen = self.parent.fen
        else:
            self.fen = INITIAL_FEN

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        self.board = chess.Board(self.fen)
        self.policy = policy
        self.children = {}  # a map from action to TreeNode
        self.action = action
        self.move = None
        self.N = 0  # visit count
        self.W = 0  # total action-value
        self.Q = 0  # mean action-value
        self.P = 0  # prior probability of selecting this node
        self.prior_p_array = None  # evaluated probabilities in this state
        self.u = 0
        self.pi = np.zeros(4096)  # updated probabilities
        self.reward = 0
        # do move to update the board state, fen
        self._do_move()

    def _do_move(self):
        if self.move is not None:
            return self.move
        elif self.action is not None:
            (from_square, to_square) = self.action
            from_square_name = chess.square_name(from_square)
            to_square_name = chess.square_name(to_square)

            piece = self.board.piece_type_at(from_square)
            promotion = False
            if piece == chess.PAWN:
                if self.board.turn == chess.WHITE and chess.square_rank(to_square) == 7:
                    promotion = True
                elif self.board.turn == chess.BLACK and chess.square_rank(to_square) == 0:
                    promotion = True

            if promotion:
                # 如果升变成马能一步杀就升变成马，否则升变成后
                knight_p = chess.Move.from_uci(from_square_name + to_square_name + 'n')
                self.board.push(knight_p)
                if self.board.is_checkmate():
                    self.move = knight_p
                else:
                    self.board.pop()    # remove last move
                    self.move = chess.Move.from_uci(from_square_name + to_square_name + 'q')
                    self.board.push(self.move)
            else:
                self.move = chess.Move.from_uci(from_square_name + to_square_name)
                self.board.push(self.move)
            self.fen = self.board.fen()  # fen after move
        return self.move

    def append_move(self, action):
        """Expand the tree node with the specified action"""
        if self.board.is_game_over(claim_draw=True):
            raise ValueError("game is over, cannot append more moves")

        sub_node = TreeNode(parent=self,
                            policy=self.policy,
                            action=action)
        self.children[action] = sub_node
        return sub_node

    def select(self, depth=1):
        """Select action among children that gives maximum action value, Q plus bonus u(P).

        The first in-tree phase of each simulation begins at the root node of the search tree,
        and finishes when the simulation reaches a leaf node sL at time-step L. At each of these
        time-steps, t < L, an action is selected according to the statistics in the search tree
        """
        target_depth = self.depth + depth
        selected_node = self
        while True:
            # go deeper
            if selected_node.is_leaf():
                selected_node.expand_with_leagl_moves()
                if selected_node.is_leaf():  # game over, nothing expanded
                    break
            (action, selected_node) = max(selected_node.children.items(), key=lambda act_node: act_node[1].get_value())
            if selected_node.depth >= target_depth:
                break

        logging.debug("node selected: %s", selected_node.get_msg())
        return selected_node

    def evaluate(self):
        """The leaf node sL is added to a queue for neural network evaluation,
        (di(p), v) = f(di(sL)), where di is a dihedral reflection or rotation selected
        uniformly at random from i between [1..8].
        Positions in the queue are evaluated by the neural network using a mini-batch size of 8;
        the search thread is locked until evaluation completes. The leaf node is expanded
        and each edge (sL, a) is initialised to
        {N(sL,a) = 0,W(sL,a) = 0,Q(sL,a) = 0,P(sL,a) = pa}; the value v is then backed up.
        """
        if self.prior_p_array is None:
            if self.policy is not None:
                features = self.get_input_features()
                output = self.policy.forward([features])
                # shape of output[0] is (1, 4096), shape of output[1] is (1, 1)
                self.prior_p_array = output[0][0]  # The first one is not rotated
                self.reward = output[1][0][0]
                logging.debug("node %s, evaluate reward %f", self.get_msg(), self.reward)
            else:
                raise ValueError("cannot evaluate if policy is None")

        return self.reward

    def play(self):
        """the child node corresponding to the played action becomes the new root node;
        the subtree below this child is retained along with all its statistics, """
        self.update_pi()
        play_node = self.select(depth=1)
        logging.debug("play node %s", play_node.get_msg())

        # Discard other child nodes
        self.children.clear()
        self.children[play_node.action] = play_node
        return play_node

    def expand_with_leagl_moves(self):
        """(Figure 2b) Expand tree by creating new children using legal moves, including PASS_MOVE.
        """
        # if self.board.is_game_over(claim_draw=True):
        #     return self

        if self.prior_p_array is None:
            # evaluate this node to get probabilities of legal moves
            self.evaluate()

        for move in self.board.generate_legal_moves():
            action = (move.from_square, move.to_square)

            sub_node = TreeNode(parent=self,
                                policy=self.policy,
                                action=action)
            self.children[action] = sub_node
            sub_node.set_prior_prob(self.prior_p_array[action[0]*64 + action[1]])

    def get_input_features(self):
        """Generate input features
        """
        return game_converter.fen_to_features(self.fen)

    def next_node(self):
        """Return the first child of this node"""
        if self.is_leaf():
            return None
        return list(self.children.values())[0]

    def set_prior_prob(self, prob: float):
        """Set the prior probability of this node"""
        add_noise = self.depth <= 10  # First 30 moves should add noise.
        if add_noise:
            self.P = 0.75 * prob + 0.25 * np.random.randint(1, 100) / 100
        else:
            self.P = prob
        self.u = self.P * C_PUCT * 0.5

    def backup(self, leaf_value):
        """(Figure 2c) The edge statistics are updated in a backward pass through each step
        t <= L.

        The visit counts are incremented, N(st, at) = N(st, at)+1
        the action-value is updated to the mean value, W(st,at) = W(st,at) + v, Q(st,at) = W/N

        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        if self.depth == 1:
            logging.debug("node backup: %s", self.get_msg())
            logging.debug("weight before update: %s", self._weights())
            logging.debug("update value: %f", leaf_value)
        # Count visit.
        self.N += 1
        # Update W
        self.W += leaf_value
        # Update Q, a running average of values for all visits.
        self.Q = self.W / self.N
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.
        if not self.is_root():
            self.u = C_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        if self.depth == 1:
            logging.debug("weight after update: %s", self._weights())

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.

        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """
        # If it is not root, this node's parent should be updated first.
        if not self.is_root():
            self.parent.update_recursive(leaf_value)
        self.backup(leaf_value)

    def feed_back_winner(self):
        """When game is over, it is then scored to give a final reward of r_T {-1,0,+1}
        The data for each time-step t is stored as (s_t,Pi_t,z_t) where z_t = ±r_T is the
        game winner from the perspective of the current player at step t
        """
        if not self.board.is_game_over(claim_draw=True):
            raise ValueError("this method can be invoked only when game is over")

        result = self.board.result(claim_draw=True)
        if result == "0-1":
            winner = chess.BLACK
        elif result == "1-0":
            winner = chess.WHITE
        elif result == "1/2-1/2":
            winner = None

        current_node = self
        while current_node is not None:
            if winner is None:
                current_node.reward = 0
            else:
                current_node.reward = 1 if winner == current_node.board.turn else -1
            current_node = current_node.parent

    def get_value(self):
        """Calculate and return the value for this node:
        this search control strategy initially prefers actions with high prior probability and
        low visit count, but asympotically prefers actions with high action-value
        """
        return self.Q + self.u

    def update_pi(self):
        total_n = 0.
        for item in self.children.items():
            (move, node) = item
            total_n += node.get_pi()

        for item in self.children.items():
            (move, node) = item
            self.pi[move[0]*64 + move[1]] = node.get_pi()/total_n

    def get_pi(self):
        """At the end of the search AlphaGo Zero selects a move a to play in the root
        position s0, proportional to its exponentiated visit count
        """
        temperature = 1  # first 30 moves
        if self.depth > 10:
            temperature = 50  # τ→0, 1/τ→a big number
        if not self.is_root():
            return pow(self.N, temperature)
        return 1

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self.children == {}

    def is_root(self):
        return self.parent is None

    def get_root(self):
        current_node = self
        while not current_node.is_root():
            current_node = current_node.parent
        return current_node

    def get_msg(self):
        if self.move is not None:
            uci_str = self.move.uci()
        else:
            uci_str = 'None'
        return "depth: %d, move: %s, turn: %d" % (self.depth, uci_str, self.board.turn)

    def _weights(self):
        return "W: %f, Q: %f, N: %d, u: %f" % (self.W, self.Q, self.N, self.u)

    def _get_pgn_game(self):
        root = self.get_root()
        game = pgn.Game()
        game.headers["Event"] = "Self play"
        next_node = root.next_node()
        game_node = game

        while True:
            game_node = game_node.add_variation(next_node.move)
            if next_node.is_leaf():
                break
            next_node = next_node.next_node()

        game.headers["Result"] = next_node.board.result(claim_draw=True)

        return game

    def save_as_pgn(self, file_path):
        game = self._get_pgn_game()
        with open(file_path, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

    def export_pgn_str(self):
        game = self._get_pgn_game()
        exporter = chess.pgn.StringExporter(headers=True)
        return game.accept(exporter)
