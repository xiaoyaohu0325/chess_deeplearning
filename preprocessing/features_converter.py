import h5py
import numpy as np
import os
import argparse
import math

batch_size = 5000


def hdf5_process(in_file, out_file):
    # initialise the output
    tmp_file = os.path.join(os.path.dirname(out_file), ".tmp." + os.path.basename(out_file))
    combined = h5py.File(tmp_file, 'w')
    try:
        fileread = h5py.File(in_file, 'r')

        features_data = fileread['features']
        actions_data = fileread['probs']
        rewards_data = fileread['rewards']

        size = len(features_data)

        combined.require_dataset(
            name='features',
            dtype=np.uint8,
            shape=(size, 8, 8, 18),
            maxshape=(None, 8, 8, 18),
            chunks=True,
            compression="lzf")
        combined.require_dataset(
            name='pi_from',
            dtype=np.float,
            shape=(size, 64),
            maxshape=(None, 64),
            chunks=True,
            compression="lzf")
        combined.require_dataset(
            name='pi_to',
            dtype=np.float,
            shape=(size, 64),
            maxshape=(None, 64),
            chunks=True,
            compression="lzf")
        combined.require_dataset(
            name='rewards',
            dtype=np.int8,
            shape=(size, 1),
            maxshape=(None, 1),
            chunks=True,
            compression="lzf")

        features = combined["features"]
        actions_from = combined["pi_from"]
        actions_to = combined["pi_to"]
        rates = combined["rewards"]

        offset = 0

        while offset < size:
            if size - offset >= batch_size:
                read_size = batch_size
            else:
                read_size = size - offset

            feature_batch = np.zeros((read_size, 8, 8, 18))
            from_batch = np.zeros((read_size, 64))
            to_batch = np.zeros((read_size, 64))
            rate_batch = np.zeros((read_size, 1))

            for i in range(read_size):
                feature_batch[i] = features_data[offset + i]
                rate_batch[i] = rewards_data[offset + i]
                probs = actions_data[offset + i]     # size is 4096
                for j in range(len(probs)):
                    if probs[j] > 0:
                        value = math.sqrt(probs[j])
                        from_square = int(j/64)
                        to_square = j % 64
                        from_batch[i, from_square] += value
                        to_batch[i, to_square] += value

                from_batch[i] = from_batch[i]/np.sum(from_batch[i])
                to_batch[i] = to_batch[i]/np.sum(to_batch[i])

            features[offset:offset+read_size] = feature_batch
            actions_from[offset:offset+read_size] = from_batch
            actions_to[offset:offset+read_size] = to_batch
            rates[offset:offset+read_size] = rate_batch

            offset += read_size
            print("percentage:", offset/size)

        fileread.close()

        combined.close()
        os.rename(tmp_file, out_file)
    except Exception as e:
        os.remove(tmp_file)
        raise e


def run_convert(cmd_line_args=None):
    """Run cancatenations. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
        description='Convert the features in hdf5 files',
        epilog="A hdf5 files is needed")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)
    parser.add_argument("--infile", "-i", help="Source HDF5 files to process", required=True)

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    hdf5_process(args.infile, args.outfile)


if __name__ == '__main__':
    run_convert()
