import math

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.datasets import mnist


# Class to facilitate the creation of two TensorFlow variables, W and b, and exposes a __call__() method to apply the
# following transformation:
# output = activation(dot(W, input) + b)
class NaiveDense:

    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = output_size
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


# Class to facilitate chaining layers by wrapping a list of layers and exposing a __call__() method that calls the
# underlying layers on the inputs, in order.
class NaiveSequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


# Class to facilitate iterating over the MNIST data in mini-batches.
class BatchGenerator:

    def __init__(self, images, labels, batch_size):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


# learning_rate = 1e-3


# def update_weights(gradients, weights):
#     for g, w in zip(gradients, weights):
#         w.assign_sub(g * learning_rate)


optimizer = optimizers.SGD(learning_rate=1e-3)


# In practice, you would almost never implement a weight update step like this by hand as shown above. Instead,
# you would use an Optimizer instance from Keras, like this:
def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


# Method to update the weights of the model after running it on one batch of data.
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# An epoch of training simply consists of repeating the training step for each batch in the training data,
# and the full training loop is simply the repetition of one epoch.
def fit(model, images, labels, epochs, batch_size):
    for epoch_counter in range(epochs):
        print(f'Epoch {epoch_counter}')
        batch_generator = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f'loss as batch {batch_counter}: {loss:.2f}')


def main():
    # Mock Keras model:
    model = NaiveSequential([
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
    ])
    assert len(model.weights) == 4

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    fit(model, train_images, train_labels, epochs=10, batch_size=128)

    predictions = model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f'accuracy: {np.mean(matches):.2f}')


main()
