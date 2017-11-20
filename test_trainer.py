from trainer import hdf5_batch_generator, shuffled_hdf5_batch_generator
import argparse
import h5py as h5
from timeit import default_timer as timer


def batch_generator(cmd_line_args=None):
    parser = argparse.ArgumentParser(description='Perform self play to generate data.')
    # required args
    parser.add_argument("data", help="Path to a hdf features file")
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 128", type=int,
                        default=128)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    dataset = h5.File(args.data, 'r')

    n_total_data = len(dataset["features"])
    # Need to make sure training data is divisible by minibatch size or get
    # warning mentioning accuracy from keras
    n_train_data = n_total_data - (n_total_data % args.minibatch)

    print("hdf5_batch_generator...")

    for i in range(100):
        start = timer()
        hdf5_batch_generator(dataset["features"],
                             dataset["pi"],
                             dataset["rewards"],
                             0,
                             n_train_data,
                             args.minibatch)
        end = timer()
        print('yield batch, elapse', end-start)

    print("shuffled_hdf5_batch_generator...")
    for i in range(100):
        start = timer()
        shuffled_hdf5_batch_generator(dataset["features"],
                                      dataset["pi"],
                                      dataset["rewards"],
                                      0,
                                      n_train_data,
                                      args.minibatch)
        end = timer()
        print('yield batch, elapse', end-start)

    dataset.close()


if __name__ == '__main__':
    batch_generator()
