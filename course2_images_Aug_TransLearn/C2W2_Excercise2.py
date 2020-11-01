import os
import random
from os import path as path
from shutil import copyfile, rmtree

# import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.optimizers import RMSprop


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, accuracy: float = None, val_acc: float = None):
        self.accuracy = accuracy
        self.val_acc = val_acc

    def on_epoch_end(self, epoch, logs=None):
        if self.accuracy and logs.get('accuracy') > self.accuracy:
            self.model.stop_training = True
            print(f'\nReached {self.accuracy * 100}% training accuracy so cancelling training!')
        if self.val_acc and logs.get('val_accuracy') > self.val_acc:
            self.model.stop_training = True
            print(f'\nReached {self.val_acc * 100}% validation accuracy so cancelling training!')


from Common.DownloadZipFile import downloadExtract

url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
filePathDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
downloadExtract(url, path.join(filePathDir, 'cats-and-dogs.zip'))
# %%
cats_v_dogs = path.join(filePathDir, 'cats-v-dogs')
# remove files and empty directory
if path.isdir(cats_v_dogs):
    rmtree(cats_v_dogs)
os.mkdir(cats_v_dogs)
os.mkdir(path.join(cats_v_dogs, 'training'))
os.mkdir(path.join(cats_v_dogs, 'training', 'cats'))
os.mkdir(path.join(cats_v_dogs, 'training', 'dogs'))

os.mkdir(path.join(cats_v_dogs, 'testing'))
os.mkdir(path.join(cats_v_dogs, 'testing', 'cats'))
os.mkdir(path.join(cats_v_dogs, 'testing', 'dogs'))
# %%
source_dir = path.join(filePathDir, 'cats-and-dogs', 'PetImages')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    total_list = os.listdir(SOURCE)
    split_len = int(len(total_list) * SPLIT_SIZE)
    training_list = random.sample(total_list, split_len)
    for i in range(len(total_list)):
        fname = total_list[i]
        fname_source = path.join(SOURCE, fname)
        if path.getsize(fname_source) > 0:
            if fname in training_list:
                fname_dest = path.join(TRAINING, fname)
            else:
                fname_dest = path.join(TESTING, fname)
            copyfile(fname_source, fname_dest)
        else:
            print(f'{fname} is zero length, so ignoring')


# YOUR CODE ENDS HERE
CAT_SOURCE_DIR = path.join(source_dir, 'Cat')
TRAINING_CATS_DIR = path.join(cats_v_dogs, 'training', 'cats')
TESTING_CATS_DIR = path.join(cats_v_dogs, 'testing', 'cats')
DOG_SOURCE_DIR = path.join(source_dir, 'Dog')
TRAINING_DOGS_DIR = path.join(cats_v_dogs, 'training', 'dogs')
TESTING_DOGS_DIR = path.join(cats_v_dogs, 'testing', 'dogs')

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# %%
print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))
# Expected output:
# 11250
# 11250
# 1250
# 1250
# %%
# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
# USE AT LEAST 3 CONVOLUTION LAYERS
model = tf.keras.models.Sequential([
    # YOUR CODE HERE
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# %%

TRAINING_DIR = path.join(cats_v_dogs, 'training')
train_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0,
                                         shear_range=0.2,
                                         horizontal_flip=True,
                                         rotaion_range=0.2,
                                         height_shift_range=0.2,
                                         width_shift_range=0.2,
                                         zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    target_size=(150, 150),
                                                    class_mode='binary')

VALIDATION_DIR = path.join(cats_v_dogs, 'testing')
validation_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              target_size=(150, 150),
                                                              class_mode='binary')
# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=224,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=25,
                    callbacks=[Callback(val_acc=0.85)]
                    )
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs
# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
plt.show()

# Desired output. Charts with training and validation metrics. No crash :)

# %%
# ------------------------------------------------
# Apply Augmentation here and plot forecast_valid
# ------------------------------------------------
TRAINING_DIR = path.join(cats_v_dogs, 'training')
train_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0,
                                         rotation_range=40,
                                         fill_mode='nearest',
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    target_size=(150, 150),
                                                    class_mode='binary')

VALIDATION_DIR = path.join(cats_v_dogs, 'testing')
validation_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              target_size=(150, 150),
                                                              class_mode='binary')
# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=224,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=25,
                    callbacks=[Callback(val_acc=0.85)]
                    )
# %%
# -----------------------------------------------------------
# Retrieve a list of list forecast_valid on training and test datasets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
plt.show()

# %%
# %%
import numpy as np
import tensorflow.keras.preprocessing.image as image

testing_images_dir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'testing_images', 'cat_vs_dog')
imgNames = os.listdir(testing_images_dir)
for fn in imgNames:
    img = image.load_img(path.join(testing_images_dir, fn), target_size=(150, 150))
    x = image.img_to_array(img) # Convert a PIL Image instance to a Numpy array
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] <= 0.5:
        print(fn + " is a dog")
    else:
        print(fn + " is a cat")
