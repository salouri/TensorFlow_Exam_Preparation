import csv
import random
from os import path

import numpy as np
import os
# force CPU use instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import requests
import tensorflow as tf
# force CPU use by tensorflow
tf.config.experimental.set_visible_devices([], 'GPU')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

embedding_dim = 100
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 160000
test_portion = .1

corpus = []

# %%
baseDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
fname1 = path.join(baseDir, 'csv_files', 'training_cleaned.csv')
print(fname1)
if not os.path.exists(fname1):
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv'
    resp = requests.get(url=url)
    print(resp.status_code)
    with open(fname1, 'wb') as f:
        f.write(resp.content)

# %%
print(os.path.exists(fname1))
num_sentences = 0
with open(fname1, encoding='utf-8', mode='r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        list_item = [row[5], 0 if row[0] == '0' else 1]
        num_sentences += 1
        corpus.append(list_item)
# %%
print(num_sentences)
print(len(corpus))
print(corpus[1])
# Expected Output:
# 1600000
# 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]
# %%
sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index)

print(vocab_size)
print(word_index['i'])
# Expected Output
# 138858
# 1
# %%
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences_padd = padded[0:split]
training_sequences_padd = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

# %%
# Note this is the 100 dimension version of GloVe dataset from Stanford
url_glove = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt'
fname2 = path.join(baseDir, 'glove.6B.100d.txt')
# !wget --no-check-certificate url_glove -O fname2 # use this code to download files on Colab notebooks
if not os.path.exists(fname2):
    resp2 = requests.get(url_glove)
    print(resp2.status_code)

    with open(fname2, 'wb') as f:
        f.write(resp2.content)

# %%
embeddings_index = {}
with open(fname2, mode='r', encoding='utf-8') as file:
    for line in file:
        values = line.split();
        word = values[0]; # each line is [word, word_coefs_vector]
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;
# %%
embeddings_matrix = np.zeros(shape=(vocab_size + 1, embedding_dim)) # this is my weights matrix
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)  # get the coefs vector of each word
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

print(len(embeddings_matrix))
# Expected Output
# 138859

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1,
                              embedding_dim,
                              input_length=max_length,
                              weights=[embeddings_matrix],
                              trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    # tf.keras.layers.MaxPooling1D(pool_size=4), # reduced accuracy a bit
    # tf.keras.layers.GlobalAveragePooling1D(), #we can't use GlobalAveragePooling with LSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# %%
num_epochs = 5

# training_padded = np.array(training_sequences_padd)
# testing_padded = np.array(test_sequences_padd)

# *************************************************************************
history = model.fit(training_sequences_padd,
                    np.array(training_labels),
                    epochs=num_epochs,
                    validation_data=(test_sequences_padd, np.array(test_labels)),
                    verbose=1)

# %%
from Common.PlotModel import plot_graphs

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
