import numpy as np
from keras import layers
from keras.datasets import mnist
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(f'\n{np.array_repr(predictions[0])}\n')
print(f'Predicted Value: {predictions[0].argmax()}')
print(f'Prediction Probability: {predictions[0][7]}')
print(f'Actual Value: {test_labels[0]}\n')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest Accuracy: {test_acc}')
