# %%
import os
import zipfile
from os import path

import requests
import tensorflow as tf

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"
filePathDir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'happy-or-sad')
fileName = filePathDir + '.zip'
# download file
print(fileName, ' ...Exists?', os.path.exists(fileName))
# %%
if not os.path.exists(fileName):
    req = requests.get(url)
    print(req.status_code)
    with open(fileName.split(os.sep)[-1], 'wb') as f:
        f.write(req.content)
        # extract file to directory
else:
    print('file already exists!!')
# %%
if os.path.exists(fileName):
    zFile = zipfile.ZipFile(fileName, 'r')
    zFile.extractall(filePathDir)
    zFile.close()
    print('file extraction complete.')
else:
    print('no zip file to extract!')

# %%
DESIRED_ACCURACY = 0.999


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs["accuracy"] >= DESIRED_ACCURACY:
            self.model.stop_training = True
            print('\nReached ', DESIRED_ACCURACY * 100, '% accuracy so cancelling training!')


callback = MyCallback()
# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

filePathDir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'happy-or-sad')
train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                   rotation_range=0.2,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(directory=filePathDir,
                                                    target_size=(150, 150),
                                                    batch_size=5,
                                                    class_mode='binary')
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    steps_per_epoch=8,
                    epochs=100,
                    verbose=1,
                    callbacks=[callback])
