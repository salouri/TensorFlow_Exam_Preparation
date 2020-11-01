# %%
import numpy as np
import tensorflow as tf

print('TensorFlow version:', tf.__version__)

# %%
# np.set_printoptions(linewidth=200)
#
# import matplotlib.pyplot as plt
# def printImg(img=None):
#     plt.imshow(img)
#     plt.show()


# %%
from Common.MyCallback import Callback


class ImgModel(Callback):
    def __init__(self):
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None
        self.model = None

    def setData(self, mydataset=None):
        if mydataset is None:
            mydataset = ""
        dataset = getattr(tf.keras.datasets, mydataset)
        # download dataset
        (self.training_images, self.training_labels), (self.test_images, self.test_labels) = dataset.load_data()
        self.training_images = self.training_images / 255.0
        self.test_images = self.test_images / 255.0
        print(self.training_images[0])

    def buildModel(self):
        print('building model...')
        labels_count = len(set(self.training_labels))
        if self.training_images is not None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dense(labels_count, activation=tf.nn.softmax)
            ])
            self.model.compile(optimizer=tf.optimizers.Adam(),  # or 'sgd'... etc
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy']
                               )
        else:
            exit('data was not loaded...end')

    def fitModel(self, epochs_count=1, myCallback=None):
        if myCallback is None:
            myCallback = Callback()
        print('Fitting for ', epochs_count, ' epochs...')
        self.model.fit(self.training_images, self.training_labels, epochs=epochs_count, callbacks=[myCallback])
        print('-----------------------------------------------------------------------------')
        print('Evaluation of the fitted model:')
        self.model.evaluate(self.test_images, self.test_labels)

    def predictLabels(self):
        print('Classify testing data...')
        classifications = self.model.predict(self.test_images)
        print('classes predicted for first item are: ', classifications[0])
        print('highest value is at index(class): ', np.where(classifications[0] == max(classifications[0]))[0][0])
        print('actual class of same item is: ', self.test_labels[0])


imgModel = ImgModel()
# %%
imgModel.setData(mydataset='fashion_mnist')
# %%
imgModel.buildModel()
# %%
# *************************************************************************
# *************************************************************************
imgModel.fitModel(10, 0.90, True)
# %%
imgModel.predictLabels()
