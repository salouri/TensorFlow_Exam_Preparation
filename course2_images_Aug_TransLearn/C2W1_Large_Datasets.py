import os
from os import path

from Common.DownloadZipFile import downloadExtract

baseDir = 'D:\\PycharmProjects\\TFD_Exam\\data'
datasetDir = path.join(baseDir, 'cats_and_dogs_filtered')
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
import matplotlib.pyplot as plt

# nrows = 4
# ncols = 4
# pic_index = 0
# # %%
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)
# pic_index += 8
# next_cat_pix = [path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index - 8: pic_index]]
# next_dog_pix = [path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index - 8: pic_index]]
#
# next_cat_dog_pix = next_cat_pix + next_dog_pix
# for i, img_path in enumerate(next_cat_dog_pix):
#     subplot = plt.subplot(nrows, ncols, i + 1)
#     subplot.axis('Off')
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
# plt.show()

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

train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

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
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()
