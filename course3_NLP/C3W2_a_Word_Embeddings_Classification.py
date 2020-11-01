## if you need to set the Tensorflow version in colab:
# %tensorflow_version 1.x
# tf.enable_eager_execution()
## install TF datasets in colab:
# !pip install -q tensorflow-datasets


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# force CPU use by tensorflow
tf.config.experimental.set_visible_devices([], 'GPU')

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# tf.keras.datasets.dataset
train_data = imdb['train']
test_data = imdb['test']
# train_data and test_data are of type: tensors
# %%

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# %%
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

#%%
sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# %%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(training_padded[3]))
print(training_sentences[3])

# %%

embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# %%
model_fname = "c3w2_a_model.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_fname, save_best_only=True)
earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# root_logdir = os.path.join(os.getcwd(), "c3w2_a_logs")
# def get_run_logdir():
#     import time_steps
#     run_id = time_steps.strftime("run_%T_%m_%d-%H_%M_%S").replace(":","_")
#     return os.path.join(root_logdir, run_id)
#
# run_logdir = get_run_logdir()
# tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
num_epochs = 100
# *************************************************************************
# *************************************************************************
model.fit(training_padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final),
          callbacks=[checkpoint_cb, earlystopping_cb],
          verbose=1)
# 6s 7ms/step - loss: 0.1418 - accuracy: 0.9531 - val_loss: 0.4792 - val_accuracy: 0.8287 ....using GlabalAveragePooling1D()
# 6s 7ms/step - loss: 1.5093e-04 - accuracy: 1.0000 - val_loss: 0.8048 - val_accuracy: 0.8262 ... using Flatten()
# 6s 7ms/step - loss: 1.0694e-04 - accuracy: 1.0000 - val_loss: 0.8365 - val_accuracy: 0.8284 ... using more units in Dense
# 6s 7ms/step - loss: 0.0032 - accuracy: 0.9991 - val_loss: 1.1135 - val_accuracy: 0.8085 ...using dropout(0.2)
# %%
model = tf.keras.models.load_model(model_fname)
# %%
model.evaluate(testing_padded, testing_labels_final)
#%%
test_sentence = "I really think this is amazing. honest."
sequence_test = tokenizer.texts_to_sequences([test_sentence])

prediction = np.argmax(model.predict(pad_sequences(sequence_test, maxlen=max_length)),axis=-1)
print(prediction)
