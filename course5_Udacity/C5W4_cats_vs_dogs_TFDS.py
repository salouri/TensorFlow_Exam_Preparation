# %%
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
# tfds.disable_progress_bar()

# Reserve 10% for validation and 10% for test
splits = ["train[:40%]", "train[40%:50%]", "train[50%:60%]"]
train_ds, validation_ds, test_ds = tfds.load("cats_vs_dogs", split=splits, as_supervised=True)
#%%
train_num_examples = sum([t for t, _ in enumerate(train_ds)])
print("Number of training samples: ", int(train_num_examples * 0.40))
valid_num_examples = sum([v for v, _ in enumerate(validation_ds)])
print("Number of validation samples: ", int(valid_num_examples * 0.40))
test_num_examples = sum([s for s, _ in enumerate(test_ds)])
print("Number of testing samples: ", int(test_num_examples * 0.40))
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis('off')
plt.show()
# %%
size = (150, 150)
batch_size = 32


def format_image(image, label):
    image = tf.image.resize(image, size) / 255.
    return image, label


train_ds = train_ds.map(format_image).cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.map(format_image).cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.map(format_image).cache().batch(batch_size).prefetch(buffer_size=10)
