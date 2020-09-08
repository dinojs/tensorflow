# Pretrained Model - Classify images of dogs and cats using a pre-trained model
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


keras = tf.keras
tfds.disable_progress_bar()

# Split the data into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',  # This dataset contains (image, label) pairs, images have different dimensions and 3 color channels
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
# %% View images
get_label_name = metadata.features['label'].int2str  # Creates a function object that we can use to get labels

for image, label in raw_test.take(2):  # Displays 2 images from the dataset
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
plt.show()  # Images are not all the same dimension

# %% Data processing - Convert all images to the same size
IMG_SIZE = 160  # All images will be resized to 160x160


def resize(image, label):
    image = tf.cast(image, tf.float32)  # cast - Convert every single pixel to be float32
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label  # Returns an image that is reshaped to IMG_SIZE


# Apply resize to every single image
train = raw_train.map(resize)
validation = raw_validation.map(resize)
test = raw_test.map(resize)

# Original image vs the new image. Height, Width, RGB
for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)

# Shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# %% Pretrained model (MobileNet V2) - trained on 1.4 million images and has 1000 different classes.
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # We'll tell the model what input shape to expect

# Create the base model from the pre-trained model MobileNet V2
# Use only its convolutional base, will not load the top layer (classification) and
# use the predetermined weights from imagenet (Googles dataset)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# Output a shape (32, 5, 5, 1280) tensor that is a feature extraction from our original (1, 160, 160, 3)
for image, _ in train_batches.take(1):
    pass
feature_batch = base_model(image)
print(feature_batch.shape)  # (32, 5, 5, 1280), 32 filters

# Freezing the Base - Disabling the training of a layer. Won’t make changes to the weights of any layers that are frozen
# We don't want to change the convolutional base that already has learned weights.
base_model.trainable = False

# %% Adding our Classifier. Fine Tuning - Adjust final layers to identify features relevant to our problem
# Instead of flattening the feature map of the base layer we will use a global average pooling layer that will average
# the entire 5x5 area of each 2D feature map and return to us a single 1280 element vector per filter.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)  # A single dense neuron as we only have two classes to predict.

#  Combining all the layers together in a model.
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

# %% Training the model - global_average_layer & prediction_layer
base_learning_rate = 0.0001  # Small learning rate to ensure that the model does not have any major changes made to it.
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Binary as we are using 2 classes
              metrics=['accuracy'])

#%% Evaluation - We can evaluate the current model prior training
initial_epochs = 3
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)  # accuracy: 0.4594

#%% Training
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)
model.save("dogs_vs_cats.h5")  # h5 - format to save models in Keras, save so no need to re-train, accuracy: 0.978

# %% Prediction
trained_model = tf.keras.models.load_model("dogs_vs_cats.h5")  # Load model

class_name = ['dog', 'cat']
predictions = trained_model.predict(validation_batches)

IMG_INDEX = 665
plt.figure()  # Visualise imagine
plt.xlabel(class_name[np.argmax(predictions[IMG_INDEX])])  # Class name
plt.imshow(validation_batches[IMG_INDEX])
plt.show()
