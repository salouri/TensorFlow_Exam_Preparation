import tensorflow as tf
import tensorflow_datasets as tfds

# %%
print("\n".join(tfds.list_builders()))
# %%
ds = tfds.load('mnist', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

# %%
ds = tfds.load('mnist', split='train')
ds = ds.take(1)  # take a single example
for example in ds:
    print(list(example.keys()))
    image = example['image']
    label = example['label']
    print(image.shape, label)
