from keras.datasets import reuters


def decode_review(review):
    word_index = reuters.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
def main():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    print(train_data[0])
    print(decode_review(train_data[0]))


main()
