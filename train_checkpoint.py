from game import Game
from player import AIPlayer, RandomPlayer
from policy import ResnetPolicy
import chess


def play(white_player, black_player, n_game=100):
    for i in range(n_game):
        game = Game(white_player, black_player)
        move = game.play()
        while move is not None:
            move = game.play()

        print('game', i, 'finished! Full moves', game.fullmove_count(), 'winner:', game.winner_color())
        game.save_to_h5("./out/checkpoint/gen0_gen1_10_128.h5")


def run_checkpoint():
    # policy_1 = ResnetPolicy.load_model("./out/model/model_10_128.json")
    # policy_1.model.load_weights("./out/random_weights_10_128")
    policy_2 = ResnetPolicy.load_model("./out/model/model_10_128.json")
    policy_2.model.load_weights("./out/train/iter1/10_128/weights.00008.hdf5")

    # assign black and white
    p1 = RandomPlayer('gen0', chess.WHITE)
    p2 = AIPlayer('gen1', chess.BLACK, policy_2, simulation=400, depth=1)
    play(p1, p2)

    # switch color
    p1 = RandomPlayer('gen0', chess.BLACK)
    p2 = AIPlayer('gen1', chess.WHITE, policy_2, simulation=400, depth=1)
    play(p2, p1)


if __name__ == '__main__':
    run_checkpoint()
