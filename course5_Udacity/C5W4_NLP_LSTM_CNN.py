# %%
import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds

# glue/sst2 has 70000 items, so might take a while to download
dataset, info = tfds.load('glue/sst2', with_info=True)
print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)
# %%
dataset_train, dataset_validation = dataset['train'], dataset['validation']
print(dataset_train)
# %%
training_reviews, training_labels = [], []
validation_reviews, validation_labels = [], []

# The dataset has 67,000 training entries, but that's a lot to process here!
# If you want to take the entire dataset: (WARNING: takes longer!!)
# for item in dataset_train.take(-1):

# take 10,000 reviews
for item in dataset_train.take(10000):
    review, label = item["sentence"], item["label"]
    training_reviews.append(str(review.numpy()))
    training_labels.append(int(label.numpy()))
print('\n Number of training reviews:', len(training_reviews))

# Get the validation data
# there's only about 800 items, so take them all
for item_v in dataset_validation.take(-1):
    review_v, label_v = item_v['sentence'].numpy(), item_v['label'].numpy()
    validation_reviews.append(str(review_v))
    validation_labels.append(int(label_v))
print("\n Number of validation reviews:", len(validation_reviews))

# %%
# There's a total of 21224 words in the reviews
# but many of them are irrelevant like with, it, of, on.
# If we take a subset of the training data, then the vocab
# will be smaller.

# A reasonable review might have about 50 words or so,
# so we can set max_length to 50 (but feel free to change it as you like)

vocab_size = 4000
embedding_dim = 16
max_length = 50  # words per review
trunc_type = 'post'
pad_type = 'post'
oov_tok = '<OOV>'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index
print(len(word_index))
# %%
training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(sequences=training_sequences, maxlen=max_length,
                                padding=pad_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation_reviews)
validation_padded = pad_sequences(sequences=validation_sequences, maxlen=max_length)

training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)


# %%
def predict_from_model(model, reviews):
    reviews_sequences = tokenizer.texts_to_sequences(reviews)
    reviews_sequences_padded = pad_sequences(sequences=reviews_sequences, maxlen=max_length,
                                             padding=pad_type, truncating=trunc_type)

    predictions = model.predict(reviews_sequences_padded)
    for i in range(len(reviews)):
        print(reviews[i])
        print(f'model_cnn Pred: {predictions[i]} ({np.round(predictions[i])})')
    return predictions


def compile_fit_model(model, lr=0.001, num_epochs=30):
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])

    history = model.fit(training_padded,
                        training_labels_final,
                        epochs=num_epochs,
                        validation_data=(validation_padded, validation_labels_final))
    return history


# %%
num_epochs = 30

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
history1 = compile_fit_model(model_cnn, 0.0001, num_epochs)
# %%
model_multi_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

history2 = compile_fit_model(model_multi_bidi_lstm, 0.0003, num_epochs)
