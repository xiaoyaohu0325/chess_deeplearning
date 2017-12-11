from keras.models import Model
from keras import layers
from keras import regularizers
from keras.optimizers import SGD
from keras import losses
from keras.models import model_from_json
import json
import os
import keras.backend as K
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

reg_control = 0.0001


class ResnetPolicy(object):

    def __init__(self, residual_blocks=19, num_cnn_filter=256, init_network=False):
        self.model = None
        self.residual_blocks = residual_blocks
        self.num_cnn_filter = num_cnn_filter

        if init_network:
            # self.__class__ refers to the subclass so that subclasses only
            # need to override create_network()
            self.model = self.create_network()
            self.run_many = self._model_forward()

    def _model_forward(self):
        """Construct a function using the current keras backend that, when given a batch
        of inputs, simply processes them forward and returns the out

        This is as opposed to model.compile(), which takes a loss function
        and training method.

        c.f. https://github.com/fchollet/keras/issues/1426
        """
        # The uses_learning_phase property is True if the model contains layers that behave
        # differently during training and testing, e.g. Dropout or BatchNormalization.
        # In these cases, K.learning_phase() is a reference to a backend variable that should
        # be set to 0 when using the network in prediction mode and is automatically set to 1
        # during training.
        if self.model.uses_learning_phase:
            forward_function = K.function([self.model.input, K.learning_phase()],
                                          self.model.outputs)

            # the forward_function returns a list of tensors
            # the first [0] gets the front tensor.
            return lambda inpt: forward_function([inpt, 0])
        else:
            # identical but without a second input argument for the learning phase
            forward_function = K.function([self.model.input], self.model.outputs)
            return lambda inpt: forward_function([inpt])

    @staticmethod
    def load_model(json_file):
        """create a new neural net object from the architecture specified in json_file
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        # create new object
        network = ResnetPolicy(init_network=False)

        network.model = model_from_json(object_specs['keras_model'])
        # if 'weights_file' in object_specs:
        #     network.model.load_weights(object_specs['weights_file'])
        network.run_many = network._model_forward()
        return network

    def save_model(self, json_file, weights_file=None):
        """write the network model to the specified file

        If a weights_file (.hdf5 extension) is also specified, model weights are also
        saved to that file and will be reloaded automatically in a call to load_model
        """
        object_specs = {
            'class': self.__class__.__name__,
            'keras_model': self.model.to_json()
        }
        # if weights_file is not None:
        #     self.model.save_weights(weights_file)
            # object_specs['weights_file'] = weights_file
        # use the json module to write object_specs to file
        with open(json_file, 'w') as f:
            json.dump(object_specs, f)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def create_network(self):
        """Create the AlphaGo Zero neural network
        """

        """
        The convolutional block applies the following modules:
        1. A convolution of 256 filters of kernel size 3*3 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        """
        inputs = layers.Input(shape=(8, 8, 119))

        # create first convolution layer of 256 filters of kernel size 3*3 with stride 1
        first_convolution_layer = layers.Conv2D(filters=self.num_cnn_filter,
                                                kernel_size=3,
                                                strides=1,
                                                padding='same',
                                                kernel_regularizer=regularizers.l2(reg_control))(inputs)
        first_convolution_layer = layers.BatchNormalization()(first_convolution_layer)
        first_convolution_layer = layers.LeakyReLU()(first_convolution_layer)

        previous_layer = first_convolution_layer

        # create residual blocks
        for i in range(self.residual_blocks):
            previous_layer = self._add_residual_blocks(previous_layer)

        # first out, a policy value of action probabilities
        policy_output = self._policy_header(previous_layer)
        # second out, a scalar value of winning
        value_output = self._value_head(previous_layer)

        return Model(inputs=inputs, outputs=[policy_output, value_output])

    def _add_residual_blocks(self, layer):
        """Create the residual block
        Each residual block applies the following modules sequentially to its input:
        1. A convolution of 256 filters of kernel size 3*3 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A convolution of 256 filters of kernel size 3*3 with stride 1
        5. Batch normalisation
        6. A skip connection that adds the input to the block
        7. A rectifier non-linearity
        """
        shortcut = layer

        # first convolution layer
        y = layers.Conv2D(filters=self.num_cnn_filter, kernel_size=3, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg_control))(layer)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        # second convolution layer
        y = layers.Conv2D(filters=self.num_cnn_filter, kernel_size=3, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg_control))(y)
        y = layers.BatchNormalization()(y)

        # A skip connection that adds the input to the block
        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

        return y

    def _policy_header(self, layer):
        """
        The policy head applies the following modules:
        1. A convolution of 2 filters of kernel size 1*1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer that outputs a vector of size 4672 corresponding to logit probabilities
        for all intersections and the pass move
        """
        y = layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg_control))(layer)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Flatten()(y)
        # give a name for the out, out dimension is 64*64
        y = layers.Dense(4672, activation="softmax", name="policy_output",
                         kernel_regularizer=regularizers.l2(reg_control))(y)

        return y

    def _value_head(self, layer):
        """
        The value head applies the following modules:
        1. A convolution of 1 filter of kernel size 1*1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer to a hidden layer of size 256
        5. A rectifier non-linearity
        6. A fully connected linear layer to a scalar
        7. A tanh non-linearity outputting a scalar in the range [ 1, 1]
        """
        y = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(reg_control))(layer)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Flatten()(y)
        y = layers.Dense(256, kernel_regularizer=regularizers.l2(reg_control))(y)
        y = layers.LeakyReLU()(y)
        y = layers.Dense(1, activation="tanh", name="value_output",
                         kernel_regularizer=regularizers.l2(reg_control))(y)

        return y

    def train(self, training_data, steps, work_dir):
        from keras.callbacks import ModelCheckpoint

        def _schedule_lrn_rate(train_step):
            """train_step equals total number of min_batch updates"""
            if 0 <= train_step < 200000:
                return 0.2
            elif 200000 <= train_step < 400000:
                return 0.02
            elif 400000 <= train_step < 600000:
                return 0.002
            else:
                return 0.0002

        mini_batch = 64
        size = len(training_data[0])
        ln_rate = _schedule_lrn_rate(steps)
        optimizer = SGD(lr=ln_rate, momentum=0.9)
        # define loss functions for each output parameter, names are set in the definition
        # of output layer.
        self.model.compile(loss={
            'policy_output': losses.categorical_crossentropy,
            'value_output': losses.mse},
            loss_weights={
                'policy_output': 1.,
                'value_output': 1.},
            optimizer=optimizer,
            metrics=["accuracy"])

        checkpoint_template = os.path.join(work_dir, "weights.{d}.hdf5".format(steps))
        checkpointer = ModelCheckpoint(checkpoint_template)

        logger.debug('Training model...')
        self.model.fit(x=training_data[0],
                       y={'policy_output': training_data[1],
                          'value_output': training_data[2]},
                       batch_size=mini_batch,
                       callbacks=[checkpointer],
                       steps_per_epoch=size//mini_batch)

