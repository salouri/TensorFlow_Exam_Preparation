import tensorflow as tf
# %%
dataset1 = tf.data.Dataset.range(10)
print(list(dataset1.as_numpy_iterator()))
# %%
dataset2 = dataset1.window(5, shift=1, drop_remainder=True)
for window in dataset2:
    print(list(window.as_numpy_iterator()))
# %%
dataset3 = dataset2.flat_map(lambda window: window.batch(5)) # window.batch converts dataset into tensor(np array)
for window in dataset3:
    print(type(window))
    for val in window:
        print(val, type(val))
# %%
dataset4 = dataset3.map(lambda wind: (wind[:-1], wind[-1:]))
for x, y in dataset4:
    print(x.numpy(), y.numpy())
#%%
dataset5 = dataset4.shuffle(buffer_size=10)
dataset6 = dataset4.batch(2, drop_remainder=True).prefetch(buffer_size=1)
for x, y in dataset6:
    print('x = ',x.numpy(),'\ny = ', y.numpy())