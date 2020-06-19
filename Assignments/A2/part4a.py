import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# New Additions
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

if "__main__" == __name__:

    # config
    batch_size = 50
    img_width, img_height, img_num_channels = 32, 32, 3
    loss_function = sparse_categorical_crossentropy
    no_classes = 100
    no_epochs = 100
    optimizer = Adam()
    validation_split = 0.2
    verbosity = 1

    # load data
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    np.load = np_load_old

    input_shape = (img_width, img_height, img_num_channels)

    # Parse numbers as floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize data
    X_train = X_train / 255
    X_test = X_test / 255

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])

    # Fit data to model
    history = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
    pred = np.argmax(model.predict(X_test), axis=1)
    
    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Visualize history
    # Plot history: Loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Validation loss history')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.show()

    # Plot history: Accuracy
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['acc'])
    plt.title('Validation accuracy history')
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('No. epoch')
    plt.show()

    # X_train = vectorize_sequences(X_train)
    # X_test = vectorize_sequences(X_test)
    # one_hot_y_train = to_categorical(y_train)
    # one_hot_y_test = to_categorical(y_test)

    # X_val = X_train[:1000]
    # partial_X_train = X_train[1000:]
    # y_val = one_hot_y_train[:1000]
    # partial_y_train = one_hot_y_train[1000:]
    # model = Sequential()

    # #TODO: input layer node count may be incorrect
    # model.add(Dense(100, activation='relu', input_shape=(10000,)))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dense(200, activation='relu'))
    # model.add(Dense(100, activation='softmax'))

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # history = model.fit(partial_X_train, partial_y_train, epochs=100, batch_size=512, validation_data=(X_val, y_val))

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']

    # plt.plot(range(100), loss, 'r', label='Training Loss')
    # plt.plot(range(100), val_loss, 'b', label='Validation Loss')
    # plt.title('Training Loss vs Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # plt.clf()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # plt.plot(range(100), acc,'r',label='Training Accuracy')
    # plt.plot(range(100), val_acc, 'b', label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # result_data = [] 