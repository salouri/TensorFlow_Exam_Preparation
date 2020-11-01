# %%
import os
from os import path

import tensorflow as tf

from Common.DownloadZipFile import downloadExtract
from Common.MyCallback import Callback

train_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
test_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
parentDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
trainPathDir: str = path.join(parentDir, 'horse-or-human')
testPathDir: str = path.join(parentDir, 'validation-horse-or-human')
urls = [train_url, test_url]
pathDirs = [trainPathDir, testPathDir]
for i in range(0, len(urls)):
    zipFilePath = pathDirs[i] + '.zip'
    # download url using requests.get()
    downloadExtract(urls[i], zipFilePath)
# %%
train_horse_dir = path.join(trainPathDir, 'horses')
train_human_dir = path.join(trainPathDir, 'humans')
test_horse_dir = path.join(testPathDir, 'horses')
test_human_dir = path.join(testPathDir, 'humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
test_horse_names = os.listdir(test_horse_dir)
test_human_names = os.listdir(test_human_dir)
print('total number of horse images: ', len(train_horse_names))
print('total number of human images: ', len(train_human_names))
print('total validation horse images:', len(test_horse_names))
print('total validation human images:', len(test_human_names))


# %%

model = tf.keras.models.Sequential()
# This is the first convolution
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
# The second convolution
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
# the third convolution
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
# # the fourth convolution
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2, 2))
# # the fifth convolution
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2, 2))
# flatten layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# %%
model.summary()

# %%
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy']
              )

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# rescalling images data
train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(trainPathDir,
                                                    target_size=(150, 150),
                                                    batch_size=150,
                                                    class_mode='binary')
test_generator = test_datagen.flow_from_directory(testPathDir,
                                                  target_size=(150, 150),
                                                  batch_size=150,
                                                  class_mode='binary')
# %%
callback = Callback(0.90)

# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    steps_per_epoch=6,
                    epochs=15,
                    verbose=1,
                    validation_data=test_generator,
                    validation_steps=1,
                    callbacks=[callback])
# %%
# import numpy as np
# # from google.colab import files
# import tensorflow.keras.preprocessing.image as image
#
# testing_images_dir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'testing_images','horse_vs_human')
# images_names = os.listdir(testing_images_dir)
# # uploaded = files.upload()
# for fn in images_names:
#     # predicting images
#     path = path.join(testing_images_dir, fn)
#     # upload an image from a path
#     img = image.load_img(path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#     print(images.shape)
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0] > 0.5:
#         print(fn + " is a human")
#     else:
#         print(fn + " is a horse")
