# Generate a play using a character predictive mode.
# Output from the previous prediction as the input for the next call to generate a sequence
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')  # Read, then decode for py2 compat.
print('Length of text: {} characters'.format(len(text)))

#%% Encoding each unique character as a different integer.
vocab = sorted(set(text))  # Sort all the unique characters in the text
char2idx = {u:i for i, u in enumerate(vocab)}  # Creating a mapping from unique characters to indices
idx2char = np.array(vocab)

def text_to_int(text):  # Transform text into its int representation
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
# print("Text:", text[:13])
# print("Encoded:", text_to_int(text[:13]))
#%% Decoding
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])
print(int_to_text(text_as_int[:13]))

#%% input: Hell | output: ello - Try to predict the last character
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)  # 101 characters for each training example

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)  # Create a stream of characters from our text data
# Turn stream of characters into batches of desired length.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)  # len of each batch, drop remaining chars (if > 100)

def split_input_target(chunk):  # For the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello
dataset = sequences.map(split_input_target)  # Apply the above function to every entry
# for x, y in dataset.take(5):
#   print("\n\nEXAMPLE\n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\nOUTPUT")
#   print(int_to_text(y))

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # Number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024
# Buffer size to shuffle the dataset, (TF data is designed to work with possibly infinite sequences, so it doesn't
# attempt to shuffle the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#%% Building the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),  # None - don't know how long the sequence will be
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,  # Return intermediate results, not only the final one
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)  # Each char will represent a probability distribution that, that char comes next.
  ])
  return model
model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

# Loss function - Predicted character by sampling the output distribution (value based on probability, not the MAX)
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#%% Compiling the Model
model.compile(optimizer='adam', loss=loss)

#%% Creating Checkpoints as it trains. This will allow us to load our model from a checkpoint and continue training it.
checkpoint_dir = './Recurrent Neural Network/training_checkpoints'  # Directory where the checkpoints will be saved
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")  # Name of the checkpoint files

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

#%% Training
history = model.fit(data, epochs=100, callbacks=[checkpoint_callback])

#%% Rebuild the model from a checkpoint using a batch_size of 1 so we can feed one peice of text to it and have a prediction.
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))  # Load latest checkpoint that stores the models weights
model.build(tf.TensorShape([1, None]))

#%% Rebuild the model from a custom checkpoint
# checkpoint_num = 10
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
# model.build(tf.TensorShape([1, None]))

#%% Generate text - Can input string of any len. Evaluation step (generating text using the learned model)
def generate_text(model, start_string):
    num_generate = 800  # Number of characters to generate

    # Converting our start string to numbers (vectorising)
    input_eval = [char2idx[s] for s in start_string]  # start_string is the input string
    input_eval = tf.expand_dims(input_eval, 0)  # [1, 2, 3] into [[1, 2, 3]] (It's expected input format)

    text_generated = []  # Empty string to store our results

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()  # Clear state of the model
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # [[1, 2, 3]] into [1, 2, 3], remove the batch dimension

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))