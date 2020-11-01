import json
import os

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 1000  # top repitive words
embidding_dim = 16  # was 16
max_length = 32  # was 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000
num_epochs = 10

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
fname = os.path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'sarcasm.json')
if not os.path.isfile(fname):
    re = requests.get(url)
    with open(fname, 'wb') as f:
        f.write(re.content)
else:
    print('file was downloaded already')
# %%
with open(fname, 'r') as file:
    datastore = json.load(file)
# %%
print(datastore[0])
sentences, labels = [], []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_size = min(training_size, len(sentences))
print(training_size)
training_sentences = np.array(sentences[0:training_size])
testing_sentences = np.array(sentences[training_size:])

training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embidding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),  # using Flatten() could crash TF on some TFDS datasets like dataset
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=1)

# %%
import matplotlib.pyplot as plt

plt.gcf()
plt.plot(history.history['accuracy'], 'r', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validating accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
# %%
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validating loss')
plt.title('Training and validation Loss')
plt.legend()
plt.figure()
plt.show()
# %%
sentence = ["granny starting to fear spiders in the garden might be real",
            "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
