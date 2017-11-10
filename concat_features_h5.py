import h5py
import numpy as np
import os
import argparse
import sys


def hdf5_cancat(hdf5_files, output):
    # initialise the output
    tmp_file = os.path.join(os.path.dirname(output), ".tmp." + os.path.basename(output))
    combined = h5py.File(tmp_file, 'w')
    try:
        combined.require_dataset(
            name='features',
            dtype=np.uint8,
            shape=(0, 8, 8, 18),
            maxshape=(None, 8, 8, 18),
            chunks=True,
            compression="gzip")
        combined.require_dataset(
            name='probs',
            dtype=np.float,
            shape=(0, 4096),
            maxshape=(None, 4096),
            chunks=True,
            compression="gzip")
        combined.require_dataset(
            name='rewards',
            dtype=np.int8,
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression="gzip")

        features = combined["features"]
        actions = combined["probs"]
        rates = combined["rewards"]

        for filename in hdf5_files:
            fileread = h5py.File(filename, 'r')
            features_data = fileread['features']
            probs_data = fileread['probs']
            rewards_data = fileread['rewards']

            start = len(features)
            end = start + len(features_data)

            features.resize((end, 8, 8, 18))
            actions.resize((end, 4096))
            rates.resize((end, 1))

            features[start:end] = features_data
            actions[start:end] = probs_data
            rates[start:end] = rewards_data

            fileread.close()

        combined.close()
        os.rename(tmp_file, output)
    except Exception as e:
        os.remove(tmp_file)
        raise e


def run_cancat(cmd_line_args=None):
    """Run cancatenations. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
        description='Cancatenate the generated hdf5 files',
        epilog="A directory containing the hdf5 files is needed")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for HDF5 files",
                        default=False, action="store_true")
    parser.add_argument("--directory", "-d", help="Directory containing HDF5 files to process")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.directory:
        if args.recurse:
            files = list(_walk_all_hdf5s(args.directory))
        else:
            files = list(_list_hdf5s(args.directory))
    else:
        files = list((f.strip() for f in sys.stdin if _is_hdf5(f)))

    hdf5_cancat(files, args.outfile)


def _is_hdf5(fname):
    return fname.strip()[-5:] == ".hdf5" or fname.strip()[-2:] == ".h5"


def _walk_all_hdf5s(root):
    """a helper function/generator to get all hdf5 files in subdirectories of root
    """
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if _is_hdf5(filename):
                # yield the full (relative) path to the file
                yield os.path.join(dirpath, filename)


def _list_hdf5s(path):
    """helper function to get all hdf5 files in a directory (does not recurse)
    """
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if _is_hdf5(f))


if __name__ == '__main__':
    run_cancat()
