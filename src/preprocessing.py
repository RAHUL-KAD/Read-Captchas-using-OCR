"""
In this file
	1. Split the dataset
	2. Map the characters to intergers and vice versa
	3. Map each image with corresponding label
	4. Visualize the image with label
"""

from src import data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Batch size for training and preprocessing
BATCH_SIZE = 16
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Mapping characters to integers
char_to_int = layers.StringLookup(
	vocabulary=list(data.characters), mask_token=None)

# Mapping integers back to original characters
int_to_char = layers.StringLookup(
	vocabulary=char_to_int.get_vocabulary(), mask_token=None, invert=True)

def split_data(images, labels, train_size=0.9, shuffle=True):
	"""
	Using this function, data will be seprated into train and validation part.

	images: Path to all the images
	labels: list of all labels
	train_size: Size into which train data will be split, default 0.9
	shuffle: To randomely shuffle the dataset

	Return:
	data samples divided into training and validation.
	"""
	# 1. Get the size of the dataset
	size = len(images)
	# 2. Make an indices array and shuffle it
	indices = np.arange(size)
	
	if shuffle:
		np.random.shuffle(indices)
	# 3. Get the size of training sample
	train_sample = int(size * train_size)
	# 4. Split the data into training and validaion steps
	x_train, y_train = images[indices[:train_sample]], labels[indices[:train_sample]]
	x_valid, y_valid = images[indices[train_sample:]], labels[indices[train_sample:]]
	return x_train, x_valid, y_train, y_valid

# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(data.images), np.array(data.labels))

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_int(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

# Now we will create dataset object
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# now we need to mp each char to number.
train_dataset = (
	train_dataset.map(
		encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
	)
	.batch(BATCH_SIZE)
	.prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# visualize the data
_, ax = plt.subplots(4, 4, figsize=(10, 5))

for batch in train_dataset.take(1):
	images = batch['image']
	labels = batch['label']
	for i in range(16):
		img = (images[i] * 255).numpy().astype('uint8')
		label = tf.strings.reduce_join(int_to_char(labels[i])).numpy().decode('utf-8')
		ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap='gray')
		ax[i // 4, i % 4].set_title(label)
		ax[i // 4, i % 4].axis('off')
plt.savefig("/ocr/Read-Captchas-using-OCR/tmp/temp.png")

plt.show()

