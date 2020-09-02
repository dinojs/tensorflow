# Neural network to classify articles of clothing
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # Load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # Split into testing and training
# train_images.shape = (60000, 28, 28) images, pixel, pixel. test_images.shape = (10000, 28, 28)
# train_images[0,23,23] - Access single pixel (194), greyscale 0 (black) - 255 (white)

# The labels are int ranging from 0-9
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data pre-processing. Scale all greyscale pixel values (0-255) to be between 0 and 1. Smaller values, easier to process
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model - architecture
model = keras.Sequential([  # Sequential = feed-forward neural network
    # input layer, flatten - Reshape the (28,28) array into a vector of 784 neurons (each pixel = 1 neuron)
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer, dense -> Each neuron from the previous layer connects to each neuron of this layer (fully-connected)
    keras.layers.Dense(128, activation='relu'),  # n of neurons, Rectify linear unit activation function

    # output layer, 10 labels, Activation function softmax -> Value for each output will be between 0 and 1
    keras.layers.Dense(10, activation='softmax')
])

# Compile the Model - Optimizer = Algorithm that performs gradient descent, loss function, output
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_images, train_labels, epochs=10)  # loss: 0.2345 - accuracy: 0.9118

# Testing
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)  # verbose - console log
print('Test accuracy:', test_acc)  # Test accuracy: 0.8810999989509583

# Prediction
predictions = model.predict(test_images)

IMG_INDEX = 5
print(labels[np.argmax(predictions[IMG_INDEX])])  # Index of the highest value in the list, Trouser (correct)

plt.figure()  # Visualise imagine
plt.imshow(test_images[IMG_INDEX])
plt.colorbar()
plt.grid(False)
plt.show()


