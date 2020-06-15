'''
    Multi-class classifier w/ neural nets
'''
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#may not need
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

if "__main__" == __name__:

    best_params = {
        'layer_count': 0,
        'node_count': 0,
        'learning_rate': 0,
        'epochs': 0,
        'loss': 10000,
        'val_loss': 10000,
        'acc': 0,
        'val_acc': 0
    }

    best_loss_params = best_params
    best_acc_params = best_params

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
    iteration = 0
    for hidden_layers in range(1, 6):
        for hidden_layer_nodes in [8, 16, 24, 32, 40, 48, 56, 64]:

                    model = Sequential()
                    model.add(Dense(64, activation='relu', input_shape=(10000,)))

                    for i in range(hidden_layers):
                        model.add(Dense(hidden_layer_nodes, activation='relu'))

                    model.add(Dense(46, activation='softmax'))

                    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

                    result_data = []
                    for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
                        print("\n### Iteration:" + str(iteration) + " hidden layers:" + str(hidden_layers) + " hidden layer nodes:"
                            + str(hidden_layer_nodes) + " learning rate:" + str(learning_rate) + " ###\n")
                        iteration += 1
                        for num_epochs in [20, 40, 60, 80, 100]:
                            K.set_value(model.optimizer.learning_rate, learning_rate)
                            history = model.fit(partial_X_train, partial_y_train, epochs=num_epochs, batch_size=512, validation_data=(X_val, y_val), verbose=0)

                            loss = history.history['loss']
                            val_loss = history.history['val_loss']
                            acc = history.history['acc']
                            val_acc = history.history['val_acc']

                            epochs = range(1, len(loss) + 1)
                            result_data.append([num_epochs, loss[-1], val_loss[-1]])
                            if((best_params['loss'] + best_params['val_loss']) > (loss[-1] + val_loss[-1]) 
                                and (best_params['acc'] + best_params['val_acc']) < (acc[-1] + val_acc[-1])):
                                best_params = {
                                    'layer_count': hidden_layers,
                                    'node_count': hidden_layer_nodes,
                                    'learning_rate': learning_rate,
                                    'epochs': num_epochs,
                                    'loss': loss[-1],
                                    'val_loss': val_loss[-1],
                                    'acc': acc[-1],
                                    'val_acc': val_acc[-1]
                                }

                            if((best_loss_params['loss'] + best_loss_params['val_loss']) > (loss[-1] + val_loss[-1])):
                                best_loss_params = {
                                    'layer_count': hidden_layers,
                                    'node_count': hidden_layer_nodes,
                                    'learning_rate': learning_rate,
                                    'epochs': num_epochs,
                                    'loss': loss[-1],
                                    'val_loss': val_loss[-1],
                                    'acc': acc[-1],
                                    'val_acc': val_acc[-1]
                                }

                            if((best_acc_params['acc'] + best_acc_params['val_acc']) < (acc[-1] + val_acc[-1])):
                                best_acc_params = {
                                    'layer_count': hidden_layers,
                                    'node_count': hidden_layer_nodes,
                                    'learning_rate': learning_rate,
                                    'epochs': num_epochs,
                                    'loss': loss[-1],
                                    'val_loss': val_loss[-1],
                                    'acc': acc[-1],
                                    'val_acc': val_acc[-1]
                                }

                            print("\n### Best Params ###")
                            print("Overall: " + str(best_params))
                            print("Accuracy: " + str(best_acc_params))
                            print("Loss: " + str(best_acc_params))

                        # plt.clf()
                        # plt.plot([item[0] for item in result_data], [item[1] for item in result_data], 'r', label='Training Loss')
                        # plt.plot([item[0] for item in result_data], [item[2] for item in result_data], 'b', label='Validation Loss')
                        # plt.title('Training Loss vs Validation Loss')
                        # plt.xlabel('Epochs')
                        # plt.ylabel('Loss')
                        # plt.legend()
                        # plt.show()
                        result_data = [] 

    # plt.clf()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # plt.plot(epochs, acc,'r',label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # results = model.evaluate(X_test, one_hot_y_test)
    # print(results)