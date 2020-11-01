# %%
import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# %%
# find available datasets
print([x for x in tfds.list_builders() if 'flowers' in x])

splits = ['train[:70%]', 'train[70%:]']
# splits = tfds.Split.All.subsplit(weighted=(70,30))
(training_set, validation_set), dataset_info = tfds.load('tf_flowers', split=splits, with_info=True, as_supervised=True)

# %%
num_classes = dataset_info.features['label'].num_classes
num_training_examples = int(0.7 * dataset_info.splits['train'].num_examples)
num_validation_examples = int(0.3 * dataset_info.splits['train'].num_examples)
print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))
# %%
for i, example in enumerate(training_set.take(5)):
    print('Image {} shape: {} label: {}'.format(i + 1, example[0].shape, example[1]))

# %%
IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.
    return image, label


BATCH_SIZE = 32
train_batches = training_set.cache().shuffle(buffer_size=num_training_examples // 4).map(format_image).batch(
    BATCH_SIZE).prefetch(1)
validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

# %%
# download a Feature Extractor using tensorflow_hub KerasLayer
# feature vector: the partial model, from TensorFlow Hub, without the final classification layer.
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
# freeze the variable in the feature extractor layer
feature_extractor.trainable = False

# %%
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes)
])
model.summary()
# %%
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

EPOCHS = 6
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    verbose=2)
# %%

class_names = np.array(dataset_info.features['label'].names)

print(class_names)
# %%
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)

print("Labels:           ", label_batch)
print("Predicted labels: ", predicted_ids)

# %%

import time

# save the model keras way:
export_path_keras = '{}.h5'.format(int(time.time()))
tf.keras.models.save_model(model, export_path_keras)
# model.save(export_path_keras) # another way to save model
# load model:
reloaded_model = tf.keras.models.load_model(export_path_keras,
                                            custom_objects={'KerasLayer': hub.KerasLayer})

## Saving and loading models in the TensorFlow SavedModel format:
### save model:
# model = tf.keras.models.clone_model(model) # this is a workaround for a bug
# tf.saved_model.save(model, export_path_keras)

### load model:
# reloaded_model= tf.saved_model.load(export_path_keras)

# Download model from Google Colab:
# !zip -r 'model.zip' {export_path_sm}
# from google.colab import files
# files.download('./mode.zip')
