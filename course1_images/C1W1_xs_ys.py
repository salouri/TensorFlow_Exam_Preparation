# %%
""""
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=kQFAr_xo0M4T
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


# %%
class DLModel:
    def __init__(self):
        self.xs = None
        self.ys = None
        self.model = None

    def setData(self, xs=None, ys=None):
        if ys is None:
            ys = []
        if xs is None:
            xs = []
        print('setting data: xs =[', str(xs), '] and ys =[', str(ys), ']')
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)

    def buildModel(self):
        print('building model...')
        self.model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def fitModel(self, epochs_count=1):
        print('fitting on the available data for ', epochs_count, ' epochs')
        self.model.fit(self.xs, self.ys, epochs=epochs_count)

    def predictModel(self, x):
        print(self.model.predict([x]))


# %%
seqModel = DLModel()
seqModel.buildModel()
# %%
seqModel.setData([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
# *************************************************************************
# *************************************************************************
seqModel.fitModel(500)
# %%
seqModel.predictModel(10.0)
