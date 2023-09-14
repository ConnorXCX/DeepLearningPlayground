import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Finding the parameters of a line neatly separating two classes of data.

# Generate each class of points by drawing their coordinates from a random distribution with a specific covariance
# matrix and a specific mean. Intuitively, the covariance matrix describes the shape of the point cloud, and the mean
# describes its position in the plane. We’ll reuse the same covariance matrix for both point clouds, but we’ll use
# two different mean values—the point clouds will have the same shape, but different positions.

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype='float32'),
                     np.ones((num_samples_per_class, 1), dtype='float32')))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# Creating the linear classification variables. A linear classifier is an affine transformation
# (prediction = W • input + b) trained to minimize the square of the difference between predictions and the targets.

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# Forward Pass Function.
def model(inputs):
    return tf.matmul(inputs, W) + b


# Mean Squared Error Loss Function.
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


learning_rate = 0.1


# Training Step Function.
def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_w, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_w * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


# Batch Training Loop.
for step in range(40):
    loss = training_step(inputs, targets)
    print(f'Loss at step {step + 1}: {loss:.4f}')

predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

# Linear Classification Model, visualized as a red line.

x = np.linspace(-1, 4, 100)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, '-r')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
