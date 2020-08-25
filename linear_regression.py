# Predict the survival rate of passengers from the titanic dataset - Linear regression
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain.head()) # Read csv into a dataframe object so can reference columns and rows
# print(dftrain.shape)  # rows, columns
# dftrain.age.hist(bins=20)  # Visulise demography
# dftrain.sex.value_counts().plot(kind='barh')  # Visulise male:female ratio
# dftrain['class'].value_counts().plot(kind='barh')  # Visualise class data (first, second, third)
# # Visualise likelihood of surviving according to sex
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# %% Convert categorical values into numeric
feature_columns = []  # Features that will be fed into the model
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Input function defines how the dataset will be converted into batches at each epoch
# epoch = number of streams of the entire dataset
# Tensorflow model requires data as "tf.data.Dataset", convert the current pandas dataframe using an input function
# Function to create an input function which is returned
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use


train_input_fn = make_input_fn(dftrain, y_train)  # Call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)  # Only 1 epoch and no shuffle during eval

# Creating the model - Linear estimator to use the linear regression algorithm
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the model
linear_est.train(train_input_fn)  # Train according to the input function
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on eval data

print(result['accuracy']) #0.7462121

#%% Access predictions by individual entry
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[1])  # Data about the person
print(y_eval.loc[1])  # If they actually survived - 1 true, 0 false
print(result[1]['probabilities'][1])  # Prediction of survival rate

# sex                          male
# age                            54
# n_siblings_spouses              0
# parch                           0
# fare                      51.8625
# class                       First
# deck                            E
# embark_town           Southampton
# alone                           y
# Name: 1, dtype: object
# 0
# 0.22543177
