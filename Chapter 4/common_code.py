import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras.datasets import reuters


# Decode imdb review back to text.
def decode_imdb_review(review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in review])


# Decode reuters newswire back to text.
def decode_reuters_newswire(newswire):
    word_index = reuters.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in newswire])


# Encode integer sequences via multi-hot encoding.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


# Embedding each label as an all-zero vector with a 1 in the place of the label index. Below method is a naive
# implementation.
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1


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
