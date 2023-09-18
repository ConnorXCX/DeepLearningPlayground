import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import imdb
from tensorflow import keras


# Decode a review back to text.
def decode_review(review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in review])


# Encode integer sequences via multi-hot encoding.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def plot_training_and_validation_loss(history_dict):
    plt.clf()
    loss_values = history_dict['loss']
    validation_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training Loss')
    plt.plot(epochs, validation_loss_values, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_training_and_validation_accuracy(history_dict):
    plt.clf()
    accuracy = history_dict['accuracy']
    validation_accuracy = history_dict['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print(decode_review(train_data[0]))

    # Vectorize data.
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Vectorize labels.
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Reference section 4.1.3 in 'Deep Learning with Python'
    model = keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Reference section 4.1.3 in 'Deep Learning with Python'
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Create a validation set by setting apart 10,000 samples from the original training data.
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # Train the model for 20 epochs (20 iterations over all samples in the training data) in mini-batches of 512
    # samples. At the same time, we will monitor loss and accuracy on the 10,000 samples that we set apart. We do so
    # by passing the validation data as the validation_data argument.
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=4,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # Initial model with 20 epochs exhibited overfitting / overoptimizing of the training data.
    plot_training_and_validation_loss(history.history)
    plot_training_and_validation_accuracy(history.history)

    results = model.evaluate(x_test, y_test)
    print(results)


main()
