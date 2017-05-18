
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

TRAIN_FILE = '/Users/ahmetkucuk/Documents/test/ar_train.tfrecord'
VALIDATION_FILE = 'ar_train.tfrecords'


def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	keys_to_features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
		'image/class/label': tf.FixedLenFeature(
			[1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
	}

	items_to_handlers = {
		'image': tfexample_decoder.Image(
			image_key='image/encoded',
			format_key='image/format',
			channels=1),
		'label': tfexample_decoder.Tensor('image/class/label'),
		'height': tfexample_decoder.Tensor('image/height'),
		'width': tfexample_decoder.Tensor('image/width')
	}

	decoder = tfexample_decoder.TFExampleDecoder(
		keys_to_features, items_to_handlers)

	image, label = decoder.decode(serialized_example, ['image', 'label'])
	#image.set_shape([256, 256, 1])
	# OPTIONAL: Could reshape into a 28x28 image and apply distortions
	# here.  Since we are not applying any distortions in this
	# example, and the next step expects the image to be flattened
	# into a vector, we don't bother.

	# # Convert from [0, 255] -> [-0.5, 0.5] floats.
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	#
	# # Convert label from a scalar uint8 tensor to an int32 scalar.
	# label = tf.cast(features['label'], tf.int32)

	return image, label


def next_batch():
	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(
			[TRAIN_FILE], num_epochs=1)

	# Even when reading in multiple threads, share the filename
	# queue.
	print(filename_queue)
	image, label = read_and_decode(filename_queue)
	return image, label

image, label = next_batch()
with tf.Session() as sess:
	im = sess.run(image)
	l = sess.run(label)
	print(im)
	print(l)