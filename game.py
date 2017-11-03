import chess


class Game(object):
    def __init__(self, white_player, black_player):
        self.players = {
            chess.WHITE: white_player,
            chess.BLACK: black_player
        }
        self.turn = chess.WHITE
        self.board = chess.Board()
        self.is_game_over = False

    def play(self):
        if self.board.is_game_over(claim_draw=True):
            return None

        fen = self.board.fen()
        move = self.players[self.turn].generate_move(fen)
        self.board.push(move)
        self.turn = not self.turn
        return move

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
