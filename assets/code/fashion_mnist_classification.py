# Import libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

# Examples and label counts of Fashion-MNIST dataset
indexes = [np.where(y_train == digit)[0][0] for digit in range(10)]
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(10, 4))
for i in range(len(indexes)):
    plt.subplot(1, 10, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()
plt.close('all')

unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels: ", dict(zip(unique, counts)))

# Data preparation
num_labels = len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

######################################
############# TODO START #############
######################################
# If you modify the code outside of this TODO block, you may receive a score of 0 points in the worst case.
# Your task is to improve the model's test score by changing the variables EPOCHS, BATCH SIZE, LEARNING RATE, and HIDDEN UNITS.
# You are allowed to edit only network parameters below.

# network parameters
EPOCHS = 250
LEARNING_RATE = 0.002
BATCH_SIZE = 32
HIDDEN_UNITS = 512

######################################
############# TODO END ###############
######################################

# Build a 3-layer MLP model
model = Sequential()
model.add(Dense(HIDDEN_UNITS, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dense(HIDDEN_UNITS))
model.add(Activation('relu'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

opt = SGD(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Summary and visualization of the model
model.summary()
plot_model(model, show_shapes=True, dpi=60)

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Test the model

# NEVER MODIFY THIS CELL
# NEVER DELETE OUTPUT OF THIS CELL

loss, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("\nTEST ACCURACY: %.2f%%" % (100.0 * acc))
