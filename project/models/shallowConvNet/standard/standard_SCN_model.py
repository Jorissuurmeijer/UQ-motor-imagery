from keras import Model, layers, constraints
from tensorflow.python.keras import backend as K

def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=22, Samples=1001, dropoutRate=0.5):
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

    input_main = layers.Input((Chans, Samples, 1))
    block1 = layers.Conv2D(40, (1, 25),  # sampling rate in used datasets is around 250,
                    # so take the values of the original paper
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=constraints.max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = layers.Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=constraints.max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = layers.Activation(square)(block1)
    block1 = layers.AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(
        block1)  # bigger pool size and strides because higher sampling rate
    block1 = layers.Activation(log)(block1)
    block1 = layers.Dropout(dropoutRate)(block1)
    flatten = layers.Flatten()(block1)
    dense = layers.Dense(nb_classes, kernel_constraint=constraints.max_norm(0.5))(flatten)
    softmax = layers.Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
