import numpy as np
import tensorflow as tf
from tensorflow import keras


# A Dense layer implemented as a Keras Layer subclass.
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__()
        self.units = units
        self.activation = activation

    # Weight Creation Function.
    def build(self, input_shape):
        input_dim = input_shape[-1]
        # add_weight() is a shortcut method for creating weights.
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer=None)

    # Forward Pass Computation.
    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


my_dense = SimpleDense(units=32, activation=tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = my_dense(input_tensor)
print(output_tensor.shape)

# model = keras.Sequential([
#     SimpleDense(32, 'relu'),
#     SimpleDense(64, 'relu'),
#     SimpleDense(32, 'relu'),
#     SimpleDense(10, 'softmax')
# ])

model = keras.Sequential([keras.layers.Dense(1)])

# Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via
# the model.compile() method.
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

inputs = np.random.random((20, 100))
targets = np.random.random((20, 100))

indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]

# To train a model, you can use the fit() method, which runs mini-batch gradient descent for you. You can also use it
# to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.
history = model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=128,
    validation_data=(val_inputs, val_targets)
)

loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)

# Once your model is trained, you use the model.predict() method to generate predictions (called inference) on new
# inputs.

# Takes a NumPy array or TensorFlow tensor and returns a TensorFlow tensor.
predictions_naive = model(inputs)

# A better way to do inference is to use the predict() method. It will iterate over the data in small batches and
# return a NumPy array of predictions. And unlike __call__(), it can also process TensorFlow Dataset objects.
predictions = model.predict(inputs, batch_size=128)
