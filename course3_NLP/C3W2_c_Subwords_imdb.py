import os
import tensorflow as tf
import tensorflow_datasets as tfds

# force CPU use instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# force CPU use by tensorflow
tf.config.experimental.set_visible_devices([], 'GPU')

# %%
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True, shuffle_files=True)
train_dataset0, test_dataset0 = dataset['train'], dataset['test']
assert isinstance(train_dataset0, tf.data.Dataset)
# %%
print(info.splits['train'].num_examples)
print(info.splits['test'].num_examples)
# %%
for input, label in train_dataset0.take(1):
    print(label)
    print(tf.shape(input))
    print(info.features['text'].ints2str(input))
    print('-------------------------------------')
    print(info.features['text'].encoder.decode(input))
# %%
tokenizer = info.features['text'].encoder
# print(tokenizer.subwords[:10])
# %%
# sample_string = 'Tensorflow, from basics to mastery'
# tokenized_string = tokenizer.encode(sample_string)
# print('Tokenized string is: {}'.format(tokenized_string))
#
# original_string = tokenizer.decode(tokenized_string)
# print('Original string is: {}'.format(original_string))
# for ts in tokenized_string:
#     print('{} ===> {}'.format(ts, tokenizer.decode([ts])))

# %%
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset1 = train_dataset0.shuffle(BUFFER_SIZE)
# test_dataset = test_dataset0.batch(BATCH_SIZE)
# .. or:
# train_dataset1 = train_dataset0.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
# test_dataset = test_dataset0.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset0))
# .. or:
train_dataset = train_dataset0.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE).prefetch(1)
test_dataset = test_dataset0.padded_batch(BATCH_SIZE)

# %%
embedding_dim = 16
print(tokenizer.vocab_size)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', recurrent_dropout=0, return_sequences=True)),
    # tf.keras.layers.LSTM(32, activation='tanh', recurrent_dropout=0, return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh', recurrent_dropout=0)),
    # tf.keras.layers.LSTM(32, activation='tanh', recurrent_dropout=0),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# *************************************************************************
# *************************************************************************
num_epochs = 10
history = model.fit(train_dataset,
                    epochs=num_epochs,
                    validation_data=test_dataset,
                    verbose=1)
# 97s 387ms/step - loss: 0.3537 - accuracy: 0.8498 - val_loss: 0.4367 - val_accuracy: 0.8093
# %%
from Common.PlotModel import plot_graphs

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
