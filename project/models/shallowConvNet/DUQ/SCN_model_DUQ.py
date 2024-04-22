import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K

from keras_uncertainty.layers import RBFClassifier
# from keras_uncertainty.layers import add_l2_regularization
import keras_uncertainty.backend


from keras.regularizers import l2


def add_l2_regularization(model, l2_strength=1e-4):
    for layer in model.layers:
        if layer.trainable_weights:
            # Wrap the regularization inside a lambda to ensure it's callable
            layer_loss = lambda: l2(l2_strength)(layer.trainable_weights[0])
            model.add_loss(layer_loss)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))


class ShallowConvNet:

    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    def build(self):

        nb_classes = 4
        Chans = 22
        Samples = 1001
        dropoutRate = 0.5

        model = keras.models.Sequential()

        # model.add(Input((Chans, Samples, 1)))

        model.add(Conv2D(40, (1, 25),        #since sampling rate is 250 i use 25 instead of 13
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2))))
        model.add(Conv2D(40, (Chans, 1), use_bias=False,
                         kernel_constraint=max_norm(2., axis=(0, 1, 2))))

        model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
        model.add(Activation(square))

        model.add(AveragePooling2D(pool_size=(1, 75), strides=(1, 15)))
        model.add(Activation(log))

        model.add(Dropout(dropoutRate))

        model.add(Flatten())
        model.add(Dense(nb_classes, kernel_constraint=max_norm(0.5)))       # 100 was nb_classes

        model.add(RBFClassifier(nb_classes, length_scale=0.2))

        # model.add(Activation('softmax'))

        optimizer = Adam(learning_rate=0.001)  # standard 0.001

        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer, metrics=["categorical_accuracy"])

        add_l2_regularization(model)

        return model      # Model(inputs=input_main, outputs=softmax)

