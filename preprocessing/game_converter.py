import numpy as np
import chess
import h5py as h5
from util.features import extract_features


def save_pgn_to_hd5(file_path, pgn, game_result):
    h5f = h5.File(file_path)

    try:
        # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
        dt = h5.special_dtype(vlen=bytes)
        if "draw" not in h5f:
            h5f.require_dataset(
                name='draw',
                dtype=dt,
                shape=(0, ),
                maxshape=(None, ),
                chunks=True,
                compression="lzf")
        if "white" not in h5f:
            h5f.require_dataset(
                name='white',
                dtype=dt,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                compression="lzf")
        if "black" not in h5f:
            h5f.require_dataset(
                name='black',
                dtype=dt,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                compression="lzf")

        if game_result == "1-0":
            dest = h5f["white"]
        elif game_result == "0-1":
            dest = h5f["black"]
        else:
            dest = h5f["draw"]

        size = len(dest)
        dest.resize((size + 1,))
        dest[size] = pgn
    except Exception as e:
        print("append to hdf5 failed")
        raise e
    finally:
        # processing complete; rename tmp_file to hdf5_file
        h5f.close()


def convert_game(game_tree):
    """Convert a game tree into nerual network input
    """
    root_node = game_tree.get_root()
    if root_node.is_leaf():
        raise ValueError("no moves available")

    # iterate moves until game end or leaf node arrived
    current_node = root_node
    while True:
        features = extract_features(current_node)
        r = current_node.reward
        pi = current_node.pi
        yield (features, pi, r)
        current_node = current_node.next_node()
        if current_node is None or current_node.is_leaf():
            break


def features_to_hd5(file_path, game_tree):
    h5f = h5.File(file_path)

    try:
        # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
        if "features" not in h5f:
            h5f.require_dataset(
                name='features',
                dtype=np.uint8,
                shape=(0, 8, 8, 18),
                maxshape=(None, 8, 8, 18),
                chunks=True,
                compression="lzf")
        if "pi" not in h5f:
            h5f.require_dataset(
                name='pi',
                dtype=np.float,
                shape=(0, 128),
                maxshape=(None, 128),
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

        for state, pi, r in convert_game(game_tree):
            features.resize((size + 1, 8, 8, 18))
            actions.resize((size + 1, 128))
            rates.resize((size + 1, 1))
            features[size] = state
            actions[size] = pi
            rates[size] = r
            size += 1

    except Exception as e:
        print("append to hdf5 failed")
        raise e
    finally:
        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
