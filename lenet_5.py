# Implementation of LeNet-5 in Python using TensorFlow Keras
# The original LeNet-5 architecture was proposed in paper
# Gradient based learning applied to document recognition by Yann LeCun et al., 1998
# Minor changes are made to the architecture regarding the Pooling layers
# which I have discussed in the notebook


import numpy as np 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import Input, layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Downloading MNIST datset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Scaling the images from 0 - 255 to 0 - 1

train_images = train_images / 255
test_images = test_images / 255



# Reshaping image vector from 3-D to a 4-D tensor

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# Zero padding 28x28 images to form 32x32

train_images = np.pad(train_images, pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values=0)
test_images = np.pad(test_images, pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values=0)


# LeNet-5 model

model = models.Sequential([
    layers.Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = "tanh", input_shape = (train_images.shape[1 : 4])),
    layers.AveragePooling2D(pool_size = 2, strides = 2, padding = "valid"),
    layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = "tanh"),
    layers.AveragePooling2D(pool_size = 2, strides = 2, padding = "valid"),
    layers.Conv2D(filters = 120, kernel_size = 5, strides = 1, activation = "tanh"),
    layers.Flatten(),
    layers.Dense(units = 84, activation = "tanh"),
    layers.Dense(units = 10, activation = "linear")

])


# Compiling model 

model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = SparseCategoricalCrossentropy(from_logits = "True"),
    metrics = ['accuracy']
)


# Training

model.fit(
    train_images, train_labels,
    epochs = 20,
    batch_size = 32
)


# Evaluating model performance

print(model.evaluate(test_images, test_labels))