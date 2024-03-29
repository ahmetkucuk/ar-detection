import numpy as np
import sys
import dataset
from tile_image import SolarImageLabeler
import tensorflow as tf
from PIL import Image
import lenet
import os

image_size = 4096
patch_size = 128
input_image_size = 128


def get_image_filenames(directory):
	image_filenames = []
	for filename in os.listdir(directory):
		if ".jpg" in filename or ".png" in filename:
			path = os.path.join(directory, filename)
			image_filenames.append([filename, path])
	return image_filenames


def preprocess(image, image_size):
	image = image.astype(np.uint8)
	image = Image.fromarray(image)
	im = image.convert("L")
	im = im.resize(size=(image_size, image_size))
	im_array = np.array(im)
	im_array = np.expand_dims(im_array, axis=2)
	return im_array


def variables():
	variables_to_restore = {}
	for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		print(var.name)
		if not ("Adam" in var.name or "beta" in var.name):
			variables_to_restore[var.name] = var
	return variables_to_restore


def label_image(imagepath, outputimagepath, session, images_placeholder, preds):

	labeler = SolarImageLabeler(imagepath, patch_size)
	patch_count = int(image_size/patch_size)

	for i in range(patch_count):
		for j in range(patch_count):
			patch_image = labeler.get_patch(i, j)
			input_image = preprocess(patch_image, input_image_size)

			preds_results = session.run(preds, feed_dict={images_placeholder: [input_image]})

			labels = ['AR', 'QS']
			label_index = preds_results.argmax(1)[0]
			label = labels[label_index]
			if label == "AR":
				labeler.add_label(i, j, label)
	labeler.save_fig(outputimagepath)


def main(args):

	images_dir = args[0]
	outputimagepath = args[1]
	model_ckpt = args[2]

	imagename_by_filepath = get_image_filenames(images_dir)

	images_placeholder = tf.placeholder("float", [None, input_image_size, input_image_size, 1])
	network_fn = lenet.lenet
	logits, end_points = network_fn(images_placeholder, num_classes=2, is_training=False)
	preds = tf.nn.softmax(logits)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
		for imagename in imagename_by_filepath:
			name = imagename[0]
			path = imagename[1]
			label_image(path, outputimagepath + name, sess, images_placeholder, preds)


if __name__ == "__main__":
	main(sys.argv[1:])

# args = ["data/20170211_001146_4096_0171.jpg", "test.png", "/Users/ahmetkucuk/Documents/log_test/"]
# args = ["data/unlabeled/", "data/labeled/", "/Users/ahmetkucuk/Documents/log_test/"]
# main(args)
# python "/home/ahmet/workspace/data/ar-detection/2013_05_10__23_59_47_34__SDO_AIA_AIA_171.png", "data/labels.png", "/home/ahmet/workspace/tensorboard/lenet/mode_100.ckpt"