# %%
import matplotlib.pyplot as plt
import numpy as np


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start: end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    # Just an arbitrary pattern, you can change it if you wish
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    # Repeats the same pattern at each period
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# %%
time_steps = np.arange(4 * 365 + 1, dtype='float32')
baseline = 10

amplitude = 40  # window of season
slope = 0.05
noise_level = 5
print(time_steps.shape)
# %%
# Create the series and update with noise
series = baseline + trend(time_steps, slope) + seasonality(time_steps, period=365, amplitude=amplitude)
series += noise(time_steps, noise_level, seed=42)
print(series.shape)

plt.figure(figsize=(10, 6))
plot_series(time_steps, series)
plt.show()
# %%
# valid_ratio = 0.25
split_time = 1000  # int(len(time_steps) * (1- valid_ratio))
time_train, x_train = time_steps[:split_time], series[:split_time]
time_valid, x_valid = time_steps[split_time:], series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()
# %% md
# Naive Forecast
# %%

naive_forecast = series[split_time - 1:-1]
naive_forecast.shape
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)
# %%
# Let's zoom in on the start of the validation period:
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)

# %%
from tensorflow import keras

print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


# %%
def moving_average_forecast(series, window_size):
    # Forecasts the mean of the last few values.
    # If window_size=1, then this is equivalent to naive forecast
    forecast = []
    for t in range(len(series) - window_size):
        series_window = series[t: t + window_size]
        forecast.append(series_window.mean())
    return np.array(forecast)


def moving_average_forecast_(series, window_size):
    # This implementation is 15 times faster!
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    forecast = mov[window_size - 1: -1] / window_size
    return forecast


# %%
import time
t1 = time.time()
moving_avg = moving_average_forecast(series, window_size=30)[split_time - 30:]
t1_ = time.time()
print(f'{round((t1_ - t1) * 100, 3)} ms')
t2 = time.time()
moving_avg2 = moving_average_forecast_(series, window_size=30)[split_time - 30:]
t2_ = time.time()
print(f'{round((t2_ - t2)* 100,3)} ms')
print('***********')
print(np.around(moving_avg[30:40], 2))
print(np.round(moving_avg2[30:40], 2))

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
# %%
# That's worse than naive forecast! The moving average does not anticipate trend or seasonality, so ' \
# let's try to remove them by using differencing. Since the seasonality period is 365 days, ..
# ..we will subtract the value at time_step "t – 365" from the value at time_step "t".

later_year_series = series[365:]
earlier_year_series = series[:-365]
diff_series = later_year_series - earlier_year_series
diff_time = time_steps[365:]
plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()
# %%
# Great, the trend and seasonality seem to be gone, so now we can use the moving average:
diff_moving_avg = moving_average_forecast(diff_series, 30)[split_time - 365 - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()
# %%
# Now let's bring back the trend and seasonality by adding the past values from t – 365:
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()
# %%

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
# %%
# Better than naive forecast, good. However the forecasts look a bit too random, because we're just adding past values,' \
# which were noisy. Let's use a moving averaging on past values to remove some of the noise:
past_values = series[split_time - 370:-359]  # shift time_steps back 365 days + 10 days for window_size
diff_moving_avg_plus_smooth_past = moving_average_forecast(past_values, window_size=11) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
