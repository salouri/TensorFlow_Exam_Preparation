import os
from os import path
import shutil
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

data_dir = 'D:\\PycharmProjects\\TFD_Exam\\data'
local_weights_file = path.join(data_dir, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,  # top layers are the output layer
                                weights=None)
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False
# %%
pre_trained_model.summary()


# %%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


# %%
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
x = tf.keras.layers.Flatten()(last_layer.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(pre_trained_model.input, x)
# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
model.summary()
# %%
# from Common.DownloadZipFile import downloadExtract
# train_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
# downloadExtract(train_url, path.join(data_dir, 'horse-or-human.zip'))
# test_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
# downloadExtract(test_url, path.join(data_dir, 'validation-horse-or-human.zip'))
#%%
# Define our example directories and files
train_dir = path.join(data_dir, 'horse-or-human', 'training')
validation_dir = path.join(data_dir, 'horse-or-human', 'validation')

train_horses_dir = path.join(train_dir, 'horses')
train_humans_dir = path.join(train_dir, 'humans')
validation_horses_dir = path.join(validation_dir, 'horses')
validation_humans_dir = path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Expected Output:
# 500
# 527
# 128
# 128
# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   shear_range=0.2)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


test_datagen = ImageDataGenerator(rescale=1. / 255.)
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=50,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.
# %%
callback = myCallback()
# *************************************************************************
# *************************************************************************
history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=10,
                    validation_data=validation_generator,
                    validation_steps=4,
                    callbacks=[callback])
# 11s 1s/step - loss: 0.0050 - accuracy: 0.9989 - val_loss: 0.0197 - val_accuracy: 0.9900
# %%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
