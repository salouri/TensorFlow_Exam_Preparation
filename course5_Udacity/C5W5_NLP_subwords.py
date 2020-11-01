
import os
# force CPU use instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import requests
import tensorflow as tf
# force CPU use by tensorflow
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

#%%
# Get the data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
print(info.splits['train'].num_examples)
print(info.splits['test'].num_examples)
#%%

tokenizer = info.features['text'].encoder

#%%
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)
#%%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    # tf.keras.layers.MaxPooling1D(pool_size=4), # lowered accuracy a bit
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),# using two bidirectional LSTM is too heavy
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True), # using LSTM alone is the worst choice in subwords
    # tf.keras.layers.Glo  balAveragePooling1D(),# not used with LSTM, same with Flatten()
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#%%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('/tmp/my_model_1.h5', save_best_only=True)

NUM_EPOCHS = 3
history = model.fit(train_dataset
                    , epochs=NUM_EPOCHS
                    , validation_data=test_dataset
                    , callbacks=[checkpoint_cb]
                    )
tf.keras.models.save_model(model,'my_model_1.h5')
model = tf.keras.models.load_model('my_model_1.h5')