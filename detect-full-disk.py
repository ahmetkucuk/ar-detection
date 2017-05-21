import numpy as np
import sys
import dataset
from tile_image import SolarImageLabeler
import tensorflow as tf
from PIL import Image
import lenet

image_size = 4096
patch_size = 512
input_image_size = 128


def preprocess(image, image_size):
	image = Image.fromarray(image)
	im = image.convert("L")
	im = im.resize(size=(image_size, image_size))
	im_array = np.array(im)
	im_array = np.expand_dims(im_array, axis=2)
	return im_array


def main(args):

	imagepath = args[0]
	outputimagepath = args[1]
	model_ckpt = args[2]
	labeler = SolarImageLabeler(imagepath, patch_size)
	patch_count = int(image_size/patch_size)

	for i in range(patch_count):
		for j in range(patch_count):
			patch_image = labeler.get_patch(i, j)
			input_image = preprocess(patch_image, image_size)

			with tf.Session() as sess:
				network_fn = lenet.lenet

				images_placeholder = tf.placeholder("float", [None, image_size, image_size, 1])
				logits, end_points = network_fn(images_placeholder, num_classes=2)
				#labels_placeholder = tf.placeholder("float", [None, 2])
				sess.run(tf.global_variables_initializer())
				saver = tf.train.Saver()
				saver.restore(sess, model_ckpt)

				preds = tf.nn.softmax(logits)
				preds_results = sess.run(preds, feed_dict={images_placeholder: input_image})
				print(preds_results)

				labels = ['AR', 'QS']
				print(labels[preds_results.argmax(1)])
				# if label != "QS":
				# 	labeler.add_label(i, j, label)
	labeler.save_fig(outputimagepath)

if __name__ == "__main__":
	main(sys.argv[1:])

# args = ["data/2013_05_10__23_59_47_34__SDO_AIA_AIA_171.png", "test.png", "/Users/ahmetkucuk/Documents/log_test/mode_2.ckpt"]
# main(args)