import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

for i in range(28):
    print(digit[i])

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[:, 7:-7, 7:-7]
my_digit = my_slice[4]

plt.imshow(my_digit, cmap=plt.cm.binary)
plt.show()
