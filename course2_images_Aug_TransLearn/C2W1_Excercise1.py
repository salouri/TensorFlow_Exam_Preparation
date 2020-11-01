import os
# hide any log messages but errors (level 1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set before importing tf
import random
import sys
import zipfile
from shutil import copyfile, rmtree

# import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def downloadExtract(url, zipFilePath: str):
    filename = zipFilePath.split(os.sep)[-1]
    if not os.path.exists(zipFilePath):
        print('Downloading file:"' + filename, '"...')

        with open(zipFilePath, 'wb') as f:
            response = requests.get(url, stream=True)
            total = response.headers.get('content-length')
            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * (downloaded / total))
                    sys.stdout.write('\r[{}{}]'.format('=' * done, '.' * (50 - done)))
                    sys.stdout.flush()
        sys.stdout.write('\n')
        print('\t---> File Download completed.')
    else:
        print('File: "' + filename + '" already exists! No download needed')
    if os.path.isfile(zipFilePath):
        print('Extracting the file....')
        zfile = zipfile.ZipFile(zipFilePath, 'r')
        zfile.extractall(zipFilePath.rstrip('.zip'))
        zfile.close()
        print('\t---> Extraction completed.')
    else:
        print('There is NO file to extract!')


url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
filePathDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
# downloadExtract(url, os.path.join(filePathDir, 'cats-and-dogs.zip'))
# %%
cats_v_dogs = os.path.join(filePathDir, 'cats-v-dogs')
# remove files and empty directory
if os.path.isdir(cats_v_dogs):
    rmtree(cats_v_dogs)
os.mkdir(cats_v_dogs)
os.mkdir(os.path.join(cats_v_dogs, 'training'))
os.mkdir(os.path.join(cats_v_dogs, 'training', 'cats'))
os.mkdir(os.path.join(cats_v_dogs, 'training', 'dogs'))

os.mkdir(os.path.join(cats_v_dogs, 'testing'))
os.mkdir(os.path.join(cats_v_dogs, 'testing', 'cats'))
os.mkdir(os.path.join(cats_v_dogs, 'testing', 'dogs'))
# %%
source_dir = os.path.join(filePathDir, 'cats-and-dogs', 'PetImages')


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    total_list = os.listdir(SOURCE)
    split_len = int(len(total_list) * SPLIT_SIZE)
    training_list = random.sample(total_list, split_len)
    for i in range(len(total_list)):
        fname = total_list[i]
        fname_source = os.path.join(SOURCE, fname)
        if os.path.getsize(fname_source) > 0:
            if fname in training_list:
                fname_dest = os.path.join(TRAINING, fname)
            else:
                fname_dest = os.path.join(TESTING, fname)
            copyfile(fname_source, fname_dest)
        else:
            print(f'{fname} is zero length, so ignoring')


# YOUR CODE ENDS HERE
CAT_SOURCE_DIR = os.path.join(source_dir, 'Cat')
TRAINING_CATS_DIR = os.path.join(cats_v_dogs, 'training', 'cats')
TESTING_CATS_DIR = os.path.join(cats_v_dogs, 'testing', 'cats')
DOG_SOURCE_DIR = os.path.join(source_dir, 'Dog')
TRAINING_DOGS_DIR = os.path.join(cats_v_dogs, 'training', 'dogs')
TESTING_DOGS_DIR = os.path.join(cats_v_dogs, 'testing', 'dogs')

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
model.compile(optimizer=RMSprop(lr=0.001), loss=' ', metrics=['accuracy'])
# %%

TRAINING_DIR = os.path.join(cats_v_dogs, 'training')
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=150,
                                                    target_size=(150, 150),
                                                    class_mode='binary')

VALIDATION_DIR = os.path.join(cats_v_dogs, 'testing')
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=50,
                                                              target_size=(150, 150),
                                                              class_mode='binary')
# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)
# 76s 338ms/step - loss: 0.0567 - accuracy: 0.9819 - val_loss: 0.7258 - val_accuracy: 0.8276
# 73s 327ms/step - loss: 0.0608 - accuracy: 0.9800 - val_loss: 0.7754 - val_accuracy: 0.8312
# 85s 378ms/step - loss: 0.0775 - accuracy: 0.9750 - val_loss: 0.9260 - val_accuracy: 0.7728
# %%
# -----------------------------------------------------------
# Retrieve a list of list forecast_valid on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
plt.show()

# Desired output. Charts with training and validation metrics. No crash :)
