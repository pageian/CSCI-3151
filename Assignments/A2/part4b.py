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
    no_epochs = 2
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

    weight_counts = [100, 200, 300, 400, 500]
    results_train = []
    results_test = []
    for weight_count in weight_counts:

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(weight_count, activation='relu'))
        model.add(Dense(weight_count, activation='relu'))
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
        results_train.append(history.history['acc'][-1])
        results_test.append(score[1])
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # # Visualize history
    # # Plot history: Loss
    # plt.plot(weight_counts, history.history['val_loss'])
    # plt.plot(weight_counts, history.history['loss'])
    # plt.title('Validation loss history')
    # plt.ylabel('Loss value')
    # plt.xlabel('No. epoch')
    # plt.show()

    # Plot history: Accuracy
    plt.plot(weight_counts, results_train)
    plt.plot(weight_counts, results_test)
    plt.title('Validation accuracy history')
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('Weights per hidden layer')
    plt.show()