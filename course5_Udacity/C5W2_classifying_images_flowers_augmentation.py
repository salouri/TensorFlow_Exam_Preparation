# %%
import glob
import logging
import numpy as np
import os
import shutil

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %%
_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_url,
                                   fname='flower_photos.tgz',
                                   extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cls in classes:
    img_path = os.path.join(base_dir, cls)
    images = glob.glob(img_path + os.sep + '*.jpg')
    print("{}: {} Images".format(cls, len(images)))
    train_split = round(len(images) * 0.8)
    train, val = images[: train_split], images[train_split:]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cls)):
            os.makedirs(os.path.join(base_dir, 'train', cls))
        shutil.move(t, os.path.join(base_dir, 'train', cls))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cls)):
            os.makedirs(os.path.join(base_dir, 'val', cls))
        shutil.move(v, os.path.join(base_dir, 'val', cls))
# %%
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# %%
batch_size = 100
IMG_SHAPE = 150

# %%
image_gen = ImageDataGenerator(rescale=1. / 255., horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(directory=train_dir, target_size=(IMG_SHAPE, IMG_SHAPE),
                                               batch_size=batch_size)


# %%
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
# %%
image_gen2 = ImageDataGenerator(rescale=1. / 255., rotation_range=45)
train_data_gen2 = image_gen2.flow_from_directory(directory=train_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 batch_size=batch_size,
                                                 class_mode='sparse')

augmented_images = [train_data_gen2[0][0][0] for i in range(5)]
plotImages(augmented_images)

# %%
image_gen3 = ImageDataGenerator(rescale=1. / 255., zoom_range=0.5)
train_data_gen3 = image_gen3.flow_from_directory(directory=train_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 batch_size=batch_size,
                                                 class_mode='sparse')

augmented_images = [train_data_gen3[0][0][0] for i in range(5)]
plotImages(augmented_images)

# %%
image_gen_train4 = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen4 = image_gen_train4.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

image_gen_val = ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')
# %%
model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5))
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
epochs = 80

history = model.fit_generator(
    train_data_gen4,
    steps_per_epoch=int(np.ceil(train_data_gen4.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(train_data_gen4.n / float(batch_size))),
    verbose=2
)
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
