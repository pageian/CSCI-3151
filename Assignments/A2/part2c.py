'''
    Multi-class classifier w/ neural nets L2 and Dropout
'''
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

def trainModel(X_train, y_train, X_val, y_val, hidden_layers = 3, hidden_nodes = 200, learning_rate = 0.001, epochs = 20, l2=False, dropout=False):

    model = Sequential()
    if dropout:
        model.add(Dropout(0.1, input_shape=(10000,)))
    else:
        model.add(Dense(64, activation='relu', input_shape=(10000,)))

    for i in range(hidden_layers):
        if l2:
            model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2()))
        else:
            model.add(Dense(hidden_nodes, activation='relu'))

    model.add(Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    K.set_value(model.optimizer.learning_rate, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=512, validation_data=(X_val, y_val), verbose = 0)

    val_loss = history.history['val_loss']

    return val_loss

if "__main__" == __name__:

    # load data
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)
    word_index = reuters.get_word_index(path="reuters_word_index.json")
    np.load = np_load_old

    X_train = vectorize_sequences(X_train)
    X_test = vectorize_sequences(X_test)
    one_hot_y_train = to_categorical(y_train)
    one_hot_y_test = to_categorical(y_test)

    X_val = X_train[:1000]
    partial_X_train = X_train[1000:]
    y_val = one_hot_y_train[:1000]
    partial_y_train = one_hot_y_train[1000:]

    color = ['b', 'r', 'y', 'g', 'o']


    loss_baseline = trainModel(partial_X_train, partial_y_train, X_val, y_val, 1, 100, 0.001, 40, False, False)
    loss_dropout = trainModel(partial_X_train, partial_y_train, X_val, y_val, 1, 100, 0.001, 40, False, True)
    loss_l2 = trainModel(partial_X_train, partial_y_train, X_val, y_val, 1, 100, 0.001, 40, True, False)
    loss_both = trainModel(partial_X_train, partial_y_train, X_val, y_val, 1, 100, 0.001, 40, True, True)

    plt.plot(range(1, 41), loss_baseline, 'b', label='Baseline')
    plt.plot(range(1, 41), loss_dropout, 'r', label='Dropout')
    plt.plot(range(1, 41), loss_l2, 'y', label='L2')
    plt.plot(range(1, 41), loss_both, 'g', label='L2 and Dropout')

    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend() 
    plt.show() 
