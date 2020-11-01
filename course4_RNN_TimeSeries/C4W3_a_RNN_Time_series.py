# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# %%
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# %%
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, 365, amplitude) + noise(time, noise_level, seed=42)

split_time = 1000
time_train, time_valid = time[:split_time], time[split_time:]
x_train, x_valid = series[:split_time], series[split_time:]


# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    window_size += 1
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# %%
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

window_size, batch_size, shuffle_buffer_size = 20, 128, 1000

train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])
# %%
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['accuracy', 'mae'])
# %%
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
history = model.fit(train_dataset, epochs=100, callbacks=[lr_schedule], verbose=1)
# %%
plt.semilogx(history.history["lr"], history.history["loss"])  # log-scaled plot on the x axis
plt.axis([1e-8, 1e-4, 0, 30])
plt.show()
# highest learning rate while stable is ~6e-6
# %%
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
# %%
optimizer = tf.keras.optimizers.SGD(lr=7e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history_new = model.fit(train_dataset, epochs=400)
# %%
forecast = []
for t in range(len(series) - window_size):
    print('.', end='')
    prediction = model.predict(series[t: t + window_size][np.newaxis])
    forecast.append(prediction)
# %%
forecast_valid = np.array(forecast[split_time - window_size:])[:, 0, 0]
# %%
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast_valid)
plt.show()
# %%
print(tf.keras.metrics.mean_absolute_error(x_valid, forecast_valid).numpy())
# %%
#-----------------------------------------------------------
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
plt.show()