import tensorflow as tf
import numpy as np
import os
from os import path
import time

# %%
# Download the Shakespeare dataset
file_abs_path = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'shakespear.txt')
download_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
path_to_file = tf.keras.utils.get_file(fname=file_abs_path, origin=download_url)
print(path_to_file)
# %%
# Read the data
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') # decode() is tf object function
print('Length of text: {} characters'.format(len(text)))
print(text[:250])
# %%
vocab = sorted(set(text))
vocab_size = len(vocab) # Length of the vocabulary in chars
# %%
# Process the text: Vectorize the text
idx2char = np.array(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
## Decode the corpus/text
text_encoded = np.array([char2idx[c] for c in text])
## Create training examples and targets
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_encoded)
# The batch method converts individual characters to sequences of the desired size.
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]#get the entire row except the last character
    target_text = chunk[1:]#get the entire row except the first one
    return input_text, target_text

dataset = sequences.map(split_input_target)
#%%
# Create training batches
BATCH_SIZE = 64
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#%%
# Build The Model


# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
model.summary()
#%%
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

EPOCHS=10

history = model.fit(dataset, epochs=EPOCHS, verbose=2)