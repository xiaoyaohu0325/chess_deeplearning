import numpy as np
import preprocessing.game_state as gamestate


def getBoard(state: gamestate):
    """
    A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    :return:
    """
    planes = np.zeros((3, 8, 8), np.int8)
    planes[0, :, :] = state.board == state.current_player  # own piece
    planes[1, :, :] = state.board == -state.current_player  # opponent stone
    planes[2, :, :] = state.board == 0
    return planes

def turnsSince(board, maximum=8):
    """
    A feature encoding the age of the piece at each location up to 'maximum'

    Note:
    - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
    - EMPTY locations are all-zero features
    :return:
    """