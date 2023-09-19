import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import boston_housing


def build_model():
    # The model ends with a single unit and no activation (it will be a linear layer). This is a typical setup
    # for scalar regression (a regression where you’re trying to predict a single continuous value).
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        # If you applied a sigmoid activation function to the last layer, the model could only learn to predict
        # values between 0 and 1. Here, because the last layer is purely linear, the model is free to learn to
        # predict values in any range.
        layers.Dense(1)
    ])
    # Note that we compile the model with the mse loss function—mean squared error, the square of the difference
    # between the predictions and the targets. This is a widely used loss function for regression problems. We’re
    # also monitoring a new metric during training: mean absolute error (MAE). It’s the absolute value of the
    # difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean
    # your predictions are off by $500 on average.
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K-fold cross-validation implementation.
# Reference section 4.3.4 in 'Deep Learning with Python'
def k_fold_cross_validation(train_data, train_targets):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    all_scores = []
    for i in range(k):
        print(f'Processing fold #{i + 1}')
        # Prepares the validation data: data from partition #k.
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        # Prepares the training data: data from all other partitions.
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        # Builds the Keras model (already compiled).
        model = build_model()
        # Trains the model (in silent mode, verbose = 0).
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=16, verbose=0)
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)
        # Evaluates the model on the validation data.
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

    # Validation MAE stops improving significantly after 120–140 epochs (this number includes the 10 epochs we
    # omitted). Past that point, we start overfitting.
    plot_validation_scores([np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)])


def plot_validation_scores(average_mae_history):
    plt.clf()
    # Omit the first 10 data points due to scaling issues / plot readability.
    truncated_mae_history = average_mae_history[10:]
    plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


# Another common type of machine learning problem is regression, which consists of predicting a continuous value
# instead of a discrete label: for instance, predicting the temperature tomorrow, given meteorological data or
# predicting the time that a software project will take to complete, given its specifications. In this example,
# we’ll attempt to predict the median price of homes in a given Boston suburb in the mid-1970s, given data points
# about the suburb at the time, such as the crime rate, the local property tax rate, and so on.
def main():
    (train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

    # Normalizing the data
    mean = train_data.mean(axis=0)
    train_data -= mean
    standard_deviation = train_data.std(axis=0)
    train_data /= standard_deviation
    test_data -= mean
    test_data -= standard_deviation

    k_fold_cross_validation(train_data, train_targets)

    # Training the final model.
    model = build_model()
    model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

    print(f'Test MAE Score: {test_mae_score}')

    predictions = model.predict(test_data)
    print(f'Model\'s guess for the sample\'s price in thousands of dollars: {predictions[0]}')


main()
