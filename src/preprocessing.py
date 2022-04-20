from src import data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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
