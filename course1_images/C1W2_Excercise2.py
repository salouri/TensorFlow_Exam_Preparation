#!/usr/bin/env python
# coding: utf-8

# ## Exercise 2
# In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing.
# There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
# 
# Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.
# 
# Some notes:
# 1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
# 2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
# 3. If you add any additional variables, make sure you use the same names as the ones used in the class
# 
# I've started the code for you below -- how would you finish it? 

# In[4]:
from os import path, getcwd

import tensorflow as tf

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'mnist.npz')
print(path)
#%%
# GRADED FUNCTION: train_mnist
def train_mnist():
    # YOUR CODE SHOULD START HERE
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') >= 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist
    # download dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    #     print(x_train[0])
    # YOUR CODE SHOULD START HERE
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    mnist_callback = MyCallback()
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # *************************************************************************
    # *************************************************************************
    # model fitting
    history = model.fit(  # YOUR CODE SHOULD START HERE
        x_train, y_train, epochs=50, callbacks=[mnist_callback]
        # YOUR CODE SHOULD END HERE
    )

    # model fitting
    return history.epoch, history.history

# %%
_, history = train_mnist()

#%%
print(history['accuracy'][-1])

