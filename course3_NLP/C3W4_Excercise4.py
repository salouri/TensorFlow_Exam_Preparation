import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as ku
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from Common.DownloadZipFile import downloadFile
from Common.PlotModel import plot_graphs

gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0],  enable=True)
except RuntimeError as e:
    print(e)

# %%
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt'
filePath = os.path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'sonnets.txt')
downloadFile(url, filePath)
# %%
data = open(filePath, mode='r', encoding='utf-8').read()
corpus = data.lower().split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])
# %%
max_sequence_len = max([len(txt) for txt in input_sequences])
input_sequences_padded = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# %%
print(max_sequence_len)
labels = input_sequences_padded[:, -1]
labels = ku.to_categorical(labels) # "one-hot encoding": use with categorical_crossentropy
predictors = input_sequences_padded[:, :-1]
print(predictors.shape)
print(labels.shape)
# %%
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1))
model.add(Dropout(0.2))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words//2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# %%
history = model.fit(x=predictors, y=labels, epochs=50, verbose=1)
# %%
plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
# %%
seed_text = "Help me Obi Wan Kenobi, you'r my only hope"
next_words = 100
reversed_word_index = {value: key for (key, value) in tokenizer.word_index.items()}



# %%
for _ in range(next_words):
    token_seq = tokenizer.texts_to_sequences([seed_text])
    token_seq_pad = pad_sequences(token_seq, padding='pre', maxlen=max_sequence_len - 1)
    predicted_word_idx = np.argmax(model.predict(np.array(token_seq_pad)), axis=-1)
    # print(predicted_word_idx)
    predicted_word = reversed_word_index.get(predicted_word_idx[0], '?')
    seed_text += " " + predicted_word
print(seed_text)
