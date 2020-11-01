import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# %%
dataset, info = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = info.features['label'].names
print("Class names: {}".format(class_names))

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


# %%

def normalize(images, labels):
    images = tf.cast(images, tf.float32)  # on non-tensor data types use: images = np.array(images).astype('float')
    images /= 255
    return images, labels


# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time_steps you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# %%
BATCH_SIZE = 32
train_dataset0 = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset0 = test_dataset.cache().batch(BATCH_SIZE)
# %%
model.fit(train_dataset0, epochs=10, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE), verbose=2)

test_loss, test_accuracy = model.evaluate(test_dataset0, steps=math.ceil(num_test_examples / 32), verbose=2)
print('model.evaluate: Accuracy on test dataset = ', test_accuracy)
# %%

for test_images, test_labels in test_dataset0.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print('predictions shape=', predictions.shape)

print('first image prediction:\n', str(predictions[0]))

print('index of maximum propability among predictions=', np.argmax(predictions[0]),
      ' ... and matching class in test_labels(for image0)=', test_labels[0])
