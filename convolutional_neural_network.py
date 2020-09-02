# Classifying 10 different objects using 60,000 32x32 color images
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()  # Load and split data

train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalise pixel values to be between 0 and 1

#%% CNN Architecture - Stack of Conv2D and MaxPooling2D layers to extract features
# The use densely connected layers to determine the class of an image based on the presence of features

# Convolutional base - Find features
model = models.Sequential()
# Will process 32 filters of 3x3 over the input data. Input shape (32, 32, 3), relu on the output of each convolution
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling using 2x2 sample and a stride of 2
# Feature map from the previous layer as input. Filters from 32 to 64.
# Data shrinks in spacial dimensions as it passed through the layers, so can afford (computationally) to add more depth.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()  # From (30, 30, 32) to (4, 4, 64). Image depth increases but the spacial dimensions decreases.

# Dense layer - Classify features
model.add(layers.Flatten())  # Make data one dimensional (all in one line, 4x4x64)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#%% Training - Using recommended hyper parameters
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#%% Evaluating the model using test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)  # accuracy: 0.6977

#%% Prediction
predictions = model.predict(test_images)

IMG_INDEX = 69
plt.figure()  # Visualise imagine
plt.xlabel(class_name[np.argmax(predictions[IMG_INDEX])])  # Class name
plt.imshow(test_images[IMG_INDEX])
plt.show()
