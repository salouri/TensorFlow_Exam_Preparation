import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt'
baseDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
fname1 = os.path.join(baseDir, 'irish-lyrics-eof.txt')
from Common.DownloadZipFile import downloadFile

downloadFile(url, fname1)
data = open(fname1).read()
corpus = data.lower().split('\n')

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # +1 is for the "<oov>" token

print(tokenizer.word_index)
print(total_words)

# %%
input_sequences = []
max_sequence_len = -1
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):  # make list of lists in the form of a PYRAMID
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

        max_sequence_len = max(max_sequence_len, len(n_gram_sequence))
# %%
# pad sequences
# max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
# input_sequences = np.array(input_sequences)

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) # one-hot encoding: use with categorical_crossentropy

# %%
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])


def encode_text(text_sequences):
    return ' '.join([reverse_word_index.get(w_seq, '?') for w_seq in text_sequences])


print(f'"{corpus[0]}"\n\t'
      f' The text sequence:\t{input_sequences[6]}\n\t'
      f' Is encoded to:  \t "{encode_text(input_sequences[6])}"')
# %%
max_length_no_labels = max_sequence_len - 1  # -1 so to not include the label column
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_length_no_labels))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

# *************************************************************************
# *************************************************************************
history = model.fit(xs, ys, epochs=50, verbose=1)
#%%
model_loc =os.path.join(os.path.dirname(os.getcwd()), 'models', os.path.basename(__file__).rstrip('.py').lower())
print(model_loc)
# save model using TF way
model.save(model_loc + '_model.h5', save_format='h5', overwrite=True)# == tf.keras.models.save_model(model, model_loc, save_format='h5')
# save model weights using TF way
model.save_weights(model_loc + '_weights.h5', save_format='h5', overwrite=True)

# %%
from Common.PlotModel import plot_graphs

plot_graphs(history, 'accuracy')

# %%
seed_text = "I've got a bad feeling about this"


def predict_next_word(seed_txt):
    token_padded = pad_sequences(tokenizer.texts_to_sequences([seed_txt]), maxlen=max_sequence_len-1, padding='pre')
    predict_it = np.argmax(model.predict(np.array(token_padded)), axis=-1)
    result = encode_text(predict_it)
    # print(seed + ' ' + result)
    return result


print(predict_next_word(seed_text))
# %%

next_words = 100
for _ in range(next_words):
    next_word_predict = predict_next_word(seed_text)
    print(next_word_predict)
    seed_text += " " + next_word_predict

print(seed_text)
