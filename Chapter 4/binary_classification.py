import numpy as np
from keras import layers
from keras.datasets import imdb
from tensorflow import keras

import common_code


# Two-class classification, or binary classification, is one of the most common kinds of machine learning problems.
# In this example, you’ll learn to classify movie reviews as positive or negative, based on the text content of the
# reviews.
def main():
    (train_data, train_labels), (test_data,
                                 test_labels) = imdb.load_data(num_words=10000)

    # Vectorize data.
    x_train = common_code.vectorize_sequences(train_data)
    x_test = common_code.vectorize_sequences(test_data)

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

    # Initial model with 20 epochs exhibited overfitting / overoptimizing of the training data after 4 epochs.
    common_code.plot_training_and_validation_loss(history.history)
    common_code.plot_training_and_validation_accuracy(history.history)

    results = model.evaluate(x_test, y_test)
    print(results)


main()
