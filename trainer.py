import numpy as np
import os
import threading
import h5py as h5
import json
from keras.optimizers import SGD
from keras import losses
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback
from policy import ResnetPolicy


#################### Now make the data generator threadsafe ####################

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(train_data, start_index, end_index, batch_size):
    # features of training data
    dataset = h5.File(train_data)
    feature_dataset = dataset["features"]
    probs_dataset = dataset["probs"]
    rewards_dataset = dataset["rewards"]
    num_batches = int((end_index - start_index)/batch_size)

    while True:
        for i in range(num_batches):
            begin = i*batch_size,
            end = (i+1)*batch_size
            yield (feature_dataset[begin:end], [probs_dataset[begin:end], rewards_dataset[begin:end]])


########## Data generator is now threadsafe and should work with multiple workers ##########

def shuffled_hdf5_batch_generator(feature_dataset,
                                  probs_dataset,
                                  rewards_dataset,
                                  indices,
                                  batch_size):
    """A generator of batches of training data for use with the fit_generator function
    of Keras. Data is accessed in the order of the given indices for shuffling.
    """
    f_batch_shape = (batch_size,) + feature_dataset.shape[1:]
    p_batch_shape = (batch_size, 4096)
    r_batch_shape = (batch_size, 1)

    f_batch = np.zeros(f_batch_shape)
    p_batch = np.zeros(p_batch_shape)
    r_batch = np.zeros(r_batch_shape)
    batch_idx = 0
    while True:
        for data_idx in indices:
            f_batch[batch_idx] = feature_dataset[data_idx]
            p_batch[batch_idx] = probs_dataset[data_idx]
            r_batch[batch_idx] = rewards_dataset[data_idx]
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (f_batch, [p_batch, r_batch])


class MetadataWriterCallback(Callback):

    def __init__(self, path):
        self.file = path
        self.metadata = {
            "epochs": [],
            "best_epoch": 0
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])

        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)


def meta_write_cb(out_directory, train_data, model, resume):
    # create metadata file and the callback object that will write to it
    meta_file = os.path.join(out_directory, "metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
    # load prior data if it already exists
    if os.path.exists(meta_file) and resume:
        with open(meta_file, "r") as f:
            meta_writer.metadata = json.load(f)
    # the MetadataWriterCallback only sets 'epoch' and 'best_epoch'. We can add
    # in anything else we like here
    # TODO - model and train_data are saved in meta_file; check that they match
    # (and make args optional when restarting?)
    meta_writer.metadata["training_data"] = train_data
    meta_writer.metadata["model_file"] = model
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    meta_writer.metadata["cmd_line_args"] = meta_writer.metadata.get("cmd_line_args", [])
    # meta_writer.metadata["cmd_line_args"].append(vars(args))

    return meta_writer


def run_training(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform training on a policy network.')
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from ResnetPolicy.save_model())")
    parser.add_argument("train_data", help="A .h5 file of training data")
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 128", type=int, default=128)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: 10", type=int, default=10)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: .01", type=float, default=.01)  # noqa: E501
    parser.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    parser.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training", nargs=3, type=float, default=[0.93, .05, .02])  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # TODO - what follows here should be refactored into a series of small functions

    resume = args.weights is not None

    # load model from json spec
    policy = ResnetPolicy.load_model(args.model)
    model = policy.model
    if resume:
        model.load_weights(os.path.join(args.out_directory, args.weights))

    # features of training data
    dataset = h5.File(args.train_data)

    n_total_data = len(dataset["features"])
    n_train_data = int(args.train_val_test[0] * n_total_data)
    # Need to make sure training data is divisible by minibatch size or get
    # warning mentioning accuracy from keras
    n_train_data = n_train_data - (n_train_data % args.minibatch)
    n_val_data = n_total_data - n_train_data
    # n_test_data = n_total_data - (n_train_data + n_val_data)

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # create metadata file and the callback object that will write to it
    meta_writer = meta_write_cb(args.out_directory, args.train_data, args.model, resume)
    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.hdf5")
    checkpointer = ModelCheckpoint(checkpoint_template)

    # load precomputed random-shuffle indices or create them
    # TODO - save each train/val/test indices separately so there's no danger of
    # changing args.train_val_test when resuming
    shuffle_file = os.path.join(args.out_directory, "shuffle.npz")
    if os.path.exists(shuffle_file) and resume:
        with open(shuffle_file, "rb") as f:
            shuffle_indices = np.load(f)
        if args.verbose:
            print("loading previous data shuffling indices")
    else:
        # create shuffled indices
        shuffle_indices = np.random.permutation(n_total_data)
        with open(shuffle_file, "wb") as f:
            np.save(f, shuffle_indices)
        if args.verbose:
            print("created new data shuffling indices")
    # training indices are the first consecutive set of shuffled indices, val
    # next, then test gets the remainder
    train_indices = shuffle_indices[0:n_train_data]
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    # test_indices = shuffle_indices[n_train_data + n_val_data:]

    # create dataset generators
    # train_data_generator = shuffled_hdf5_batch_generator(
    #     dataset["features"],
    #     dataset["probs"],
    #     dataset["rewards"],
    #     train_indices,
    #     args.minibatch)
    # val_data_generator = shuffled_hdf5_batch_generator(
    #     dataset["features"],
    #     dataset["probs"],
    #     dataset["rewards"],
    #     val_indices,
    #     args.minibatch)
    train_data_generator = batch_generator(
        args.train_data,
        0,
        n_train_data,
        args.minibatch)
    val_data_generator = batch_generator(
        args.train_data,
        n_train_data,
        n_train_data + n_val_data,
        args.minibatch)

    optimizer = SGD(lr=args.learning_rate, decay=args.decay, momentum=0.9)

    # define loss functions for each output parameter, names are set in the definition
    # of output layer.
    model.compile(loss={
        'policy_output': losses.categorical_crossentropy,
        'value_output': losses.mse},
        loss_weights={'policy_output': 1., 'value_output': 1.},
        optimizer=optimizer,
        metrics=["accuracy"])

    samples_per_epoch = (args.epoch_length or n_train_data)

    if args.verbose:
        print("STARTING TRAINING")

    model.fit_generator(
        generator=train_data_generator,
        steps_per_epoch=int(samples_per_epoch/args.minibatch),
        epochs=args.epochs,
        callbacks=[checkpointer, meta_writer],
        validation_data=val_data_generator,
        validation_steps=int(n_val_data/args.minibatch))


if __name__ == '__main__':
    run_training()
