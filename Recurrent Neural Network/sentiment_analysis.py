# Sentiment Analysis of IMDB movie review dataset from keras
from keras.datasets import imdb  # 25 000 labelled reviews from IMDB (positive or negative)
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
#%% Data processing - reviews must all have the same len.
# If len(review) > 250 then trim off the extra words, if len(review) < 250 add 0's to make it equal to 250.
train_data = sequence.pad_sequences(train_data, MAXLEN)  # Keras in-built function
test_data = sequence.pad_sequences(test_data, MAXLEN)

#%% Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),  # Word embedding layer, 32 is the output vector dimension
    tf.keras.layers.LSTM(32),  # LSTM (Long-Short Term Memory) layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # Value between 0, 1
])
#%% Training
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])  # Binary as only 2 output expected
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)  # Use 20% of the training data to evaluate

results = model.evaluate(test_data, test_labels)
print(results)  # acc: 0.8506

#%% Encode our reviews so the network can understand it
word_index = imdb.get_word_index()  # Get all words indexes
def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  # Check if a word exists in the vocabulary, replace it with its index, or replace it with 0 if unknown
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]  # padding. Return a list of lists
# text = "that movie was just amazing, so amazing"
# encoded = encode_text(text)
# print(encoded)
#%% Decode
reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PADDING = 0
    text = ""
    for num in integers:
        if num != PADDING:
            text += reverse_word_index[num] + " "
    return text[:-1]  # Return everything but the last space
# print(decode_integers(encoded))

#%% Prediction
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1, 250))
  pred[0] = encoded_text
  result = model.predict(pred)
  print(result[0])

positive_review = "That movie was amazing, really loved it and would definately watch it again because it was amazingly great"
predict(positive_review)  # [0.9837942] - Positive

negative_review = "This movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)  # [0.35992023] - Negative