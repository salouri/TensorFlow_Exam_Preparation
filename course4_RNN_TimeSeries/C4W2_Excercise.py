import numpy as np
import tensorflow as tf

print(tf.__version__)
# %%
from course4_RNN_TimeSeries.C4W1_Time_Series_Patterns import plot_series, trend, seasonality, noise


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # pull from the full dataset a small amount (buffer_size) and select from it randomly
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + noise(time, noise_level)

split_time = 1000
time_train = time[: split_time]
time_valid = time[split_time:]
x_train = series[:split_time]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

plot_series(time, series)
# %%

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[window_size]),
    tf.keras.layers.Dense(32, 'relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=5e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=1)

# %%

forecast = []
for time in range(len(series) - window_size):
    prediction = model.predict(series[time: time + window_size][None])
    forecast.append(prediction)

forecast_valid = forecast[split_time - window_size:]
results = np.array(forecast_valid)[:, 0, 0]

# %%
plot_series(time_valid, x_valid)
plot_series(time_valid, results)

# %%
mae = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
print(mae)
