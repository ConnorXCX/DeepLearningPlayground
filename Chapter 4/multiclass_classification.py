from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical
from tensorflow import keras

import common_code


# In this example, we build a model to classify Reuters newswires into 46 mutually exclusive topics. Because we have
# many classes, this problem is an instance of multiclass classification, and because each data point should be
# classified into only one category, the problem is more specifically an instance of single-label multiclass
# classification. If each data point could belong to multiple categories (in this case, topics), we’d be facing a
# multilabel multiclass classification problem.
def main():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    # Vectorize data.
    x_train = common_code.vectorize_sequences(train_data)
    x_test = common_code.vectorize_sequences(test_data)

    # Vectorize labels. Keras built-in one-hot encoding method, replacing the above to_one_hot method.
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    # Reference section 4.2.3 in 'Deep Learning with Python'
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(46, activation='softmax')
    ])

    # Reference section 4.2.3 in 'Deep Learning with Python'
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a validation set by setting apart 1,000 samples from the original training data.
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    # Train the model for 20 epochs (20 iterations over all samples in the training data) in mini-batches of 512
    # samples. At the same time, we will monitor loss and accuracy on the 1,000 samples that we set apart. We do so
    # by passing the validation data as the validation_data argument.
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=9,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # Initial model with 20 epochs exhibited overfitting / overoptimizing of the training data after 9 epochs.
    common_code.plot_training_and_validation_loss(history.history)
    common_code.plot_training_and_validation_accuracy(history.history)

    results = model.evaluate(x_test, y_test)
    print(results)


main()
