# %%
import numpy as np
import tensorflow as tf

print(tf.__version__)

# %%
from course4_RNN_TimeSeries.C4W1_Time_Series_Patterns import trend, seasonality, noise, plot_series


def windowed_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # pull from the full dataset a small amount (buffer_size) and select from it randomly
    dataset = dataset.cache().repeat().shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# %%
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
# %%
split_time = 2000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
# %%
window_size = 20
batch_size = 32
shuffle_buffer = 1000

dataset_train = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)
dataset_valid = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer)
# layer_0 = tf.keras.layers.Dense(units=1, input_shape=[window_size], activation='relu')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1))
# %%
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=7e-6, momentum=0.09),
              metrics=['accuracy', 'mae'])  # SGD: Stochastic Gradient Decent
#%%
# set a different lr on each epoch based on it's number
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
model.fit(dataset_train,
          epochs=500,
          steps_per_epoch=split_time // batch_size,
          validation_data=dataset_valid,
          verbose=2,
          callbacks=[lr_schedule, early_stopping])
# %%
# my_layer = model.layers[0]
# print('Layer weights {}'.format(my_layer.get_weights()))
# %%
forecast = []

for time in range(len(series) - window_size):
    prediction = model.predict(series[time: time + window_size][np.newaxis])  # "np.newaxis" == None
    forecast.append(prediction)

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]
# %%
plot_series(time_valid, x_valid)
plot_series(time_valid, results)

# %%
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
