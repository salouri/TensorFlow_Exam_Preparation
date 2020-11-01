# %%
import csv
import os

# hide any log messages but errors (level 1)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # set before importing tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Common.DownloadZipFile import downloadFile

# force CPU use instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# force CPU use by tensorflow
# tf.config.experimental.set_visible_devices([], 'GPU')


# %%
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    forecast = model.predict(ds)
    return forecast


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

baseDir = os.path.join('D:\\PycharmProjects\\TFD_Exam\\data', 'csv_files')
print(baseDir)
file_path = os.path.join(baseDir, "daily-min-temperatures.csv")
file_name = file_path.split(os.sep)[-1]
downloadFile(url, file_path)
# %%
time_step, temps = [], []

with open(file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    step = 1
    for row in reader:
        temps.append(float(row[1]))
        time_step.append(step)
        step = step + 1

series = np.array(temps)
time = np.array(time_step)
plt.figure(figsize=(10, 6))
plot_series(time, series)

split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train,
                             window_size=60,
                             batch_size=100,
                             shuffle_buffer=shuffle_buffer_size)
valid_set = windowed_dataset(x_valid,
                             window_size=60,
                             batch_size=100,
                             shuffle_buffer=shuffle_buffer_size)
print(x_train.shape, x_valid.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule], verbose=1)

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])
plt.show()
# %%
print('***************************')
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

optimizer = tf.keras.optimizers.SGD(lr=2e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model_fname = "c4w4_exercise.h5"
# save model using checkpoint_cbcallback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_fname, save_best_only=True)
# another way:  model.save('name_model.h5', save_format='h5', overwrite=True)
earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(train_set,
                    epochs=500,
                    validation_data=valid_set,
                    callbacks=[checkpoint_cb, earlystopping_cb],
                    verbose=2)

model = tf.keras.models.load_model(model_fname)
# %%
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()
print(tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())
