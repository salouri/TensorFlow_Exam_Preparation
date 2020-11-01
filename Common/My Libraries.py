import os

abs_path = ''
os.getcwd()
os.sep
os.listdir(abs_path)
os.mkdir(abs_path)
os.pathsep
os.path
os.path.dirname(abs_path)
os.path.join(abs_path, 'fname')
os.path.exists(abs_path)
os.path.isdir(abs_path)
os.path.isfile(abs_path)
os.path.getsize(abs_path)

# -------------------------------------------------------------
import sys

done = 20
sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
sys.stdout.flush()  # send buffer to terminal
# -------------------------------------------------------------

import io

out_file = io.open('fname.ext', 'w', encoding='utf-8')
out_file.write('\t strings \n')
out_file.close()

# -------------------------------------------------------------
import numpy as np

myList = []
np.array(myList, dtype=float)
np.set_printoptions(linewidth=200)
np.arange(1, 10, dtype=float)  # get a range of floats
np.where(myList >= 5)  # get the np.array from myList where condition is met
np.array_split(myList, 28)  # 28 dimensions(rows): 28 * 28
np.array(myList).astype(dtype='float')
np.array(myList).append('val')
myList = np.expand_dims(myList, axis=0)  # or axis=-1
np.unique(myList)
np.loadtxt('filename', delimiter=',', skiprows=1)
np.reshape(myList, (-1, 28, 28, 1))  # -1 means automatically set this dimension

# -------------------------------------------------------------
import requests

req = requests.get('url')
with open('fname' + abs_path, 'wb') as f:
    f.write(req.content)
# -------------------------------------------------------------

import zipfile

zfile = zipfile.ZipFile(abs_path + 'zip_file_path', 'r')
zfile.extractall(abs_path + 'dir_path')
zfile.close()
# -------------------------------------------------------------

import shutil

shutil.rmtree(abs_path)
shutil.copyfile('fname_path_source', 'fname_path_dest')

# -------------------------------------------------------------

import tensorflow as tf

tf.keras.callbacks
tf.keras.callbacks.Callback
tf.keras.datasets
tf.keras.datasets.fashion_mnist
tf.keras.datasets.mnist
tf.keras.models
tf.keras.models.Model
tf.keras.models.Sequential
tf.keras.optimizers
tf.keras.optimizers.RMSprop(lr=0.001)

tf.keras.preprocessing.image.ImageDataGenerator
img = tf.keras.preprocessing.image.load_img(abs_path, target_size=(150, 150))
tf.keras.preprocessing.image.img_to_array(img=img)
# +++++++++++++++++++++++++++++++

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token="<OOV>")  # == tokenizer
tokenizer.fit_on_texts(sentences=['numpy array sentences'])
tokenizer.word_index
sequences = tokenizer.texts_to_sequences()  # == sequences
padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences)
# +++++++++++++++++++++++++++++++
### To save your model from Colab notebook

# from google.colab import drive
# drive.mount('/content/drive')
# model.save('/content/drive/My Drive/Colab Notebooks/literature_model2.h5')
# drive.flush_and_unmount()
# -------------------------------------------------------------
import matplotlib.pyplot as plt

epochs, acc, val_acc = []
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()

# -------------------------------------------------------------

import json

file = open('fname', 'r')
data = json.load(file)
data = json.loads('string')
