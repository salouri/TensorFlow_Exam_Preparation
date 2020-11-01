import os
from os import path
# hide any log messages but errors (level 1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # set before importing tf
from Common.DownloadZipFile import downloadExtract

datasetDir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'cats_and_dogs_filtered')
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
downloadExtract(url, datasetDir + '.zip')
# %%
train_dir = path.join(datasetDir, 'train')
validation_dir = path.join(datasetDir, 'validation')

train_cats_dir = path.join(train_dir, 'cats')
train_dogs_dir = path.join(train_dir, 'dogs')

validation_cats_dir = path.join(validation_dir, 'cats')
validation_dogs_dir = path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))

print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

# %%
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the forecast_valid to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
# %%
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=RMSprop(lr=0.001))
# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
test_generator = test_datagen.flow_from_directory(validation_dir,
                                                  batch_size=20,
                                                  target_size=(150, 150),
                                                  class_mode='binary')
# %%
from Common.MyCallback import Callback

# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=100,
                    validation_steps=50,
                    epochs=15,
                    callbacks=[Callback(val_acc=0.75)],
                    verbose=2)

# %%
# Plot training and validation accuracy per epoch
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))  # Get number of epochs

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()
# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.figure()
plt.show()
# %%
# ------------------------------------------------
# Apply Augmentation to above code
# ------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
test_generator = test_datagen.flow_from_directory(validation_dir,
                                                  batch_size=20,
                                                  target_size=(150, 150),
                                                  class_mode='binary')
# %%
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=100,
                    validation_steps=50,
                    epochs=15,
                    callbacks=[Callback(val_acc=0.75)],
                    verbose=1)
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))  # Get number of epochs
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()
# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.figure()
plt.show()

# %%
import numpy as np
import tensorflow.keras.preprocessing.image as image

testing_images_dir = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'testing_images', 'cat_vs_dog')
imgNames = os.listdir(testing_images_dir)
for fn in imgNames:
    img = image.load_img(path.join(testing_images_dir, fn), target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] <= 0.5:
        print(fn + " is a dog")
    else:
        print(fn + " is a cat")
