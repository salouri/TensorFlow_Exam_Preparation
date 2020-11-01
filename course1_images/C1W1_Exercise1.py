import numpy as np
import tensorflow as tf
from tensorflow import keras

# %%
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error)
# %%
xs = np.arange(1, 7, dtype=float)  # bedroom numbers
ys = np.array([50 + 50 * x for x in xs])/100

print(xs)
print(ys)
# %%
# *************************************************************************
# *************************************************************************
model.fit(xs, ys, epochs=1000)
# %%
print(model.predict([7.0]))
exit('Thank you')
