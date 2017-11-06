from game import Game
from player import AIPlayer
from policy import ResnetPolicy
import chess


def play(white_player, black_player, n_game=100):
    for i in range(n_game):
        game = Game(white_player, black_player)
        move = game.play()
        while move is not None:
            move = game.play()

        print('game', i, 'finished! Full moves', game.fullmove_count(), 'winner:', game.winner_color())
        game.save_to_h5("./out/checkpoint/gen0_gen1.h5")


def run_checkpoint():
    policy_1 = ResnetPolicy.load_model("./out/model.json")
    policy_1.model.load_weights("./out/random_weights")
    policy_2 = ResnetPolicy.load_model("./out/model.json")
    policy_2.model.load_weights("./out/train/weights.00000.hdf5")

    # assign black and white
    p1 = AIPlayer('gen0', chess.WHITE, policy_1, simulation=40, depth=2)
    p2 = AIPlayer('gen1', chess.BLACK, policy_2, simulation=40, depth=2)
    play(p1, p2)

    # switch color
    p1 = AIPlayer('gen0', chess.BLACK, policy_1, simulation=40, depth=2)
    p2 = AIPlayer('gen1', chess.WHITE, policy_2, simulation=40, depth=2)
    play(p2, p1)


if __name__ == '__main__':
    run_checkpoint()
