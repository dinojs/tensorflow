# Classify flowers
import tensorflow as tf
import pandas as pd

# 3 different classes of species: Setosa, Versicolor, Virginic
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Information about each flower is the following: sepal length, sepal width, petal length, petal width
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# Save to csv from http link
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Use species as label
train_y = train.pop("Species")
test_y = test.pop("Species")

def input_fn(features, labels, training=True, batch_size=256):  # Input function
    # Convert input to dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if training
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Inputs that will be fed to the model
my_feature_columns = []
for key in train.keys():  # train.keys to retrieve columns
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Building the model: Deep Neural Network or LinearClassifier for classification. DNN better as data might not have a linear correspondence
classifier = tf.estimator.DNNClassifier( # estimator - Module that stores pre-made modules, DNN is one of those
    feature_columns=my_feature_columns,
    hidden_units=[30, 10], # DNN with 2 hidden layers with 30 and 10 neurons
    n_classes=len(SPECIES)) # 3 classes

#%% Training
# lambda - Create function in one line. Used so we can return the function object
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

#%% Evaluate
result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print(result["accuracy"])  # 0.8666667

#%% Prediction
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

predict = {  # Test data
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for flower in predictions:
    class_id = flower['class_ids'][0]  # Class prediction
    probability = flower["probabilities"][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))
# expected = ['Setosa', 'Versicolor', 'Virginica']
# Prediction is "Setosa" (62.6%)
# Prediction is "Versicolor" (43.0%)
# Prediction is "Virginica" (54.2%)


# INPUT FROM THE USER
# predict = {}
# print("Please type numeric values as promoted.")
# for feature in features:
#     valid = True
#     while valid:
#         val = input(feature + ": ")  # Wait for input
#         if not val.isdigit(): valid = False
#
#     predict[feature] = [float(val)]  # Must be a list even if only one input

# SepalLength: 1.5
# SepalWidth: 4.3
# PetalLength: 1.6
# PetalWidth: 2.5
# Prediction is "Virginica"(70.8 %)


