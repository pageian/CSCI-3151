import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

if "__main__" == __name__:

    # load data
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    np.load = np_load_old

    X_train = vectorize_sequences(X_train)
    X_test = vectorize_sequences(X_test)
    one_hot_y_train = to_categorical(y_train)
    one_hot_y_test = to_categorical(y_test)

    X_val = X_train[:1000]
    partial_X_train = X_train[1000:]
    y_val = one_hot_y_train[:1000]
    partial_y_train = one_hot_y_train[1000:]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10000,)))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    result_data = []
    for num_epochs in [20, 40, 60, 80, 100]:
        history = model.fit(partial_X_train, partial_y_train, epochs=num_epochs, batch_size=512, validation_data=(X_val, y_val))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        result_data.append([num_epochs, loss[-1], val_loss[-1], acc[-1], val_acc[-1]])

    plt.plot([item[0] for item in result_data], [item[1] for item in result_data], 'r', label='Training Loss')
    plt.plot([item[0] for item in result_data], [item[2] for item in result_data], 'b', label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot([item[0] for item in result_data], [item[3] for item in result_data],'r',label='Training Accuracy')
    plt.plot([item[0] for item in result_data], [item[4] for item in result_data], 'b', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    result_data = [] 