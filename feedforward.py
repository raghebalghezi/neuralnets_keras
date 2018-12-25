from typing import Tuple, Dict

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    #deep Model
    model1 = Sequential()
    model1.add(Dense(7,activation='relu',input_shape=(n_inputs,), init='uniform'))
    model1.add(Dense(7,activation='relu', init='uniform'))
    model1.add(Dense(n_outputs,activation='linear'))
    model1.compile(optimizer="adam",loss='mse', metrics=['mae'])
    #wide model
    model2 = Sequential()
    model2.add(Dense(14,activation='relu',input_shape=(n_inputs,), init='uniform'))
    model2.add(Dense(n_outputs,activation='linear'))
    model2.compile(optimizer="adam",loss='mse', metrics=['mae'])

    return model1, model2



def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    #relu
    model1 = Sequential()
    model1.add(Dense(32,activation='relu',input_shape=(n_inputs,), init='uniform'))
    model1.add(Dense(16,activation='relu', init='uniform'))
    model1.add(Dense(n_outputs,activation='sigmoid'))
    model1.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])


    #tanh
    model2 = Sequential()
    model2.add(Dense(32,activation='tanh',input_shape=(n_inputs,), init='uniform'))
    model2.add(Dense(16,activation='tanh', init='uniform'))
    model2.add(Dense(n_outputs,activation='sigmoid'))
    model2.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])


    return model1, model2


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    #drop-out
    model1 = Sequential()
    model1.add(Dense(16,activation='relu',input_shape=(n_inputs,), init='uniform'))
    model1.add(Dropout(0.01))
    model1.add(Dense(n_outputs,activation='softmax'))
    model1.compile(optimizer="adam",loss='categorical_crossentropy', metrics=['accuracy'])

    #No drop-out
    model2 = Sequential()
    model2.add(Dense(16,activation='relu',input_shape=(n_inputs,), init='uniform'))
    model2.add(Dense(n_outputs,activation='softmax'))
    model2.compile(optimizer="adam",loss='categorical_crossentropy', metrics=['accuracy'])

    return model1, model2

def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Dict, Model, Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
        )
    """
    #Early Stopping
    model1 = Sequential()
    model1.add(Dense(64,activation='sigmoid',input_shape=(n_inputs,), init='uniform'))
    model1.add(Dense(32,activation='tanh', init='uniform'))
    model1.add(Dense(16,activation='tanh', init='uniform'))
    model1.add(Dense(n_outputs,activation='sigmoid'))
    model1.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping_1 = EarlyStopping(monitor='loss', patience=10)
    parm1 = {"epochs":50, "batch_size": 32,"callbacks":[early_stopping_1]}
    

    #Non-Early Stopping
    model2 = Sequential()
    model2.add(Dense(64,activation='sigmoid',input_shape=(n_inputs,), init='uniform'))
    model2.add(Dense(32,activation='tanh', init='uniform'))
    model2.add(Dense(16,activation='tanh', init='uniform'))
    model2.add(Dense(n_outputs,activation='sigmoid'))
    model2.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])
    parm2 = {"epochs":50, "batch_size": 32}

    
    return model1,parm1, model2, parm2
