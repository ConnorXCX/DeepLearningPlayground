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
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])
