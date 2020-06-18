'''
    Multi-class classifier w/ neural nets
'''
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def graphData(y, xlabel):

    plt.clf()
    plt.plot(range(1, 21), [item[0] for item in y][0], 'b', label='Training Loss')
    plt.plot(range(1, 21), [item[1] for item in y][0], 'r', label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()  

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

def trainModel(X_train, y_train, X_val, y_val, hidden_layers = 3, hidden_nodes = 200, learning_rate = 0.001, epochs = 20):

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10000,)))

    for i in range(hidden_layers):
        model.add(Dense(hidden_nodes, activation='relu'))

    model.add(Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    K.set_value(model.optimizer.learning_rate, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=512, validation_data=(X_val, y_val), verbose = 0)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return loss, val_loss

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

    epochs_data =[]
    loss, loss_val = trainModel(partial_X_train, partial_y_train, X_val, y_val, epochs=20)
    epochs_data.append([loss, loss_val])

    graphData(epochs_data, 'epochs')
