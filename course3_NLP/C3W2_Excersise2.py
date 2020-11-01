import csv
import os
from os import path

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
baseDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
fname1 = path.join(baseDir, 'csv_files', 'bbc-text.csv')
if not path.exists(fname1):
    url_bbc = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
    req = requests.get(url_bbc)
    with open(fname1, 'wb') as file:
        file.write(req.content)
else:
    print('file {0} exists already'.format(fname1.split(os.sep)[-1]))
# %%

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]
print(len(stopwords))
# Expected Output
# 153
# %%
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'
training_portion = .8
# %%
sentences = []
labels = []

with open(fname1, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        labels.append(row[0])
        sentence = row[1]
        # remove stopwords from sentences
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)
#%%
print(len(labels))
print(len(sentences))
# Expected Output
# 2225
# 2225
# tv future hands viewers home theatre systems  plasma high-definition tvs  digital
# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------- training data/labels - validation data/labels
train_size = int(training_portion * len(sentences))
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]
validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))
# Expected output (if training_portion=.8)
# 1780
# 1780
# 1780
# 445
# 445
# %% ---------------------------------------------------------------------------- train_padded , validation_padded
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))
# Expected Ouput
# 449
# 120
# 200
# 120
# 192
# 120
# %%
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

# Expected output
# 445
# (445, 120)
# %%  ----------------------------------------------- train_labels_padded , validation_labels_padded
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
training_labels_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_labels_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
print(training_labels_seq[0])
print(training_labels_seq[1])
print(training_labels_seq[2])
print(training_labels_seq.shape)

print(validation_labels_seq[0])
print(validation_labels_seq[1])
print(validation_labels_seq[2])
print(validation_labels_seq.shape)

# Expected output
# [4]
# [2]
# [1]
# (1780, 1)
# [5]
# [4]
# [3]
# (445, 1)
# %%
labels_count = len(set(labels))
print(f'labels count: {labels_count}')
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# %%

num_epochs = 30
# *************************************************************************
# *************************************************************************
history = model.fit(train_padded, training_labels_seq, epochs=num_epochs, verbose=1,
                    validation_data=(validation_padded, validation_labels_seq))

# %%
from Common.PlotModel import plot_graphs

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# %%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

# Expected output
# (1000, 16)
# %%
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
