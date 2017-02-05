from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import load_mnist
from models import mlp
import numpy as np
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 20

def train():
    X_train, y_train, X_test, y_test = load_mnist()

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = mlp()

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train()