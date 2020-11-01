# %%
import numpy as np
import tensorflow as tf

from Common.MyCallback import Callback

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# %%
labels_count = len(set(training_labels))
if training_images is not None:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(labels_count, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),  # or 'sgd'... etc
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.summary(line_length=150)
else:
    exit('data was not loaded...end')
# %%
epochs_count = 20
# *************************************************************************
# *************************************************************************
history = model.fit(training_images, training_labels, epochs=epochs_count, callbacks=[Callback()])
# %%
print(history.epoch[-1])
print(history.history['accuracy'][-1])
# %%
print('-----------------------------------------------------------------------------')
print('Evaluation of the fitted model:')
test_acc, test_loss = model.evaluate(test_images, test_labels)

print('Classify testing data...')
classifications = model.predict(test_images)
print('classes predicted for first item are: ', classifications[0])
print('highest value is at index(class): ', np.where(classifications[0] == max(classifications[0]))[0][0])
print('actual class of same item is: ', test_labels[0])

# %%

#
# exit()
