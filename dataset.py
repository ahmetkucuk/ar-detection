import os
import tensorflow as tf
from scipy import misc
import numpy as np
from PIL import Image

from sklearn.utils import shuffle


def get_image(imagename):
	temp = Image.open(imagename)
	keep = temp.copy()
	temp.close()
	return keep

def _get_filenames_and_classes(dataset_dir):
	"""Returns a list of filenames and inferred class names.

	Args:
	  dataset_dir: A directory containing a set of subdirectories representing
		class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
	  A list of image file paths, relative to `dataset_dir` and the list of
	  subdirectories, representing class names.
	"""
	directories = []
	class_names = []
	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			if ".png" in filename:
				path = os.path.join(directory, filename)
				photo_filenames.append(path)

	return photo_filenames, sorted(class_names)


def get_datasets(dataset_dir, image_size):

	filenames, class_names = _get_filenames_and_classes(dataset_dir=dataset_dir)
	data = []
	labels = []
	for i in filenames:
		data.append(get_image(i))
		if "/AR/" in i:
			labels.append([1, 0])
		else:
			labels.append([0, 1])
	print("Finished Reading images")
	data = preprocess(data, image_size)
	data, labels = shuffle(data, labels)
	split_at = int(len(data) * 0.8)
	train_files = data[:split_at]
	train_labels = labels[:split_at]

	val_files = data[split_at:]
	val_labels = labels[split_at:]
	
	return ARDataset(data=train_files, labels=train_labels), ARDataset(data=val_files, labels=val_labels)


def preprocess(images, image_size):

	processed_images = []
	for i in images:
		im = i.convert("L")
		im = im.resize(size=(image_size, image_size))
		im_array = np.array(im)
		im_array = np.expand_dims(im_array, axis=2)
		processed_images.append(im_array)
	return processed_images


class ARDataset(object):

	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.batch_index = 0

	def next_batch(self, batch_size):
		if self.batch_index*batch_size + batch_size > len(self.data):
			self.batch_index = 0
		batched_data, batched_labels = self.data[self.batch_index*batch_size: self.batch_index*batch_size + batch_size], self.labels[self.batch_index*batch_size: self.batch_index*batch_size + batch_size]
		self.batch_index += 1
		return batched_data, batched_labels

	def size(self):
		return len(self.data)
