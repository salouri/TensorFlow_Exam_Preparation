import os
from os import path

import tensorflow as tf

url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# req = requests.get(url)
# with open(local_weights_file, 'wb') as f:
#     f.write(req.content)
data_dir = 'D:\\PycharmProjects\\TFD_Exam\\data'
local_weights_file = path.join(data_dir, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
# %%
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=local_weights_file)
# pre_trained_model.load_weights(local_weights_file)
# %%
# Freezing (by setting layer.trainable = False) prevents the weights in a given layer ...
# from being updated during training.
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
# %%
from tensorflow.keras.optimizers import RMSprop

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(pre_trained_model.input, x)
# %%
model.compile(optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'],
              loss='binary_crossentropy')
# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

datasetDir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'cats_and_dogs_filtered')
train_dir = path.join(datasetDir, 'train')
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=40,
                                                    target_size=(150, 150),
                                                    class_mode='binary')

validation_dir = path.join(datasetDir, 'validation')
validation_datagen = ImageDataGenerator(rescale=1. / 255.)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=40,
                                                              class_mode='binary',
                                                              target_size=(150, 150))
# %%
from Common.MyCallback import Callback

# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    steps_per_epoch=50,
                    validation_steps=25,
                    verbose=1,
                    callbacks=[Callback(accuracy=0.98, val_acc=0.98)])
# %%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.gcf()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
