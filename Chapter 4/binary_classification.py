import numpy as np
from keras.datasets import imdb


# Decode a review back to text.
def decode_review(coded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    return ''.join(
        [reverse_word_index.get(i - 3, "?") for i in coded_review])


# Encode integer sequences via multi-hot encoding.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Vectorize data.
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Vectorize labels.
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')


main()
