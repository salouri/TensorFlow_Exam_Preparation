import tensorflow as tf


# YOUR CODE STARTS HERE
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] >= 0.998:
            self.model.stop_training = True
            print('\nReached 99.8% accuracy so cancelling training!')


# %%
# YOUR CODE ENDS HERE

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# %%
# YOUR CODE STARTS HERE
training_images = training_images.reshape(-1, 28, 28, 1) / 255.0 # because MNIST images come in vectors
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
labels_count = len(set(test_labels))
# YOUR CODE ENDS HERE
# %%
model = tf.keras.models.Sequential([
    # YOUR CODE STARTS HERE
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(labels_count, activation=tf.keras.activations.softmax)
    # YOUR CODE ENDS HERE
])
# %%
# YOUR CODE STARTS HERE
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])
# %%
# *************************************************************************
# *************************************************************************
model.fit(training_images, training_labels, epochs=20, callbacks=[Callback()])
# %%
eval_acc, eval_loss = model.evaluate(test_images, test_labels)
print('Evaluation Accuracy: ', round(eval_acc, 2), ' Evaluation Loss:', round(eval_loss, 2))
# %%
import numpy as np
classifs = model.predict(test_images)
img_predictions = classifs[90]
print(type(img_predictions))
print(np.argmax(img_predictions, axis=-1))
print(img_predictions.tolist().index(img_predictions.max()))
print(test_labels[90])
# YOUR CODE ENDS HERE
# %%
