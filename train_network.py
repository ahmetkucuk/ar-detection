import alexnet
import tensorflow as tf
from dataset import get_datasets
import sys


def train(args):
	if len(args) < 5:
		print("arguments not valid: 0-> dataset_dir, 1->log_dir, 2-> learning_rate, 3-> image_size, 4-> batch_size, 5-> epoches")
		exit()

	dataset_dir = args[0]
	log_dir = args[1]
	learning_rate = float(args[2])
	image_size = int(args[3])
	batch_size = int(args[4])
	epoches = int(args[5])

	optimizer = tf.train.AdamOptimizer(learning_rate)
	tf.logging.set_verbosity(tf.logging.INFO)

	network_fn = alexnet.alexnet_v2
	images_placeholder = tf.placeholder("float", [None, image_size, image_size, 1])
	labels_placeholder = tf.placeholder("float", [None, 2])

	logits, end_points = network_fn(images_placeholder)
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))
	with tf.name_scope("evaluations"):
		tf.summary.scalar('loss', cross_entropy)

	train_step = optimizer.minimize(cross_entropy)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels_placeholder, 1), tf.argmax(logits, 1)), tf.float32))
	with tf.name_scope("evaluations"):
		tf.summary.scalar('accuracy', accuracy)


	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

		sess.run(tf.global_variables_initializer())
		dataset, validation_dataset = get_datasets(dataset_dir=dataset_dir, image_size=image_size)
		n_of_patches = dataset.size()
		iter = 0
		epoch_count = 0

		while (iter * batch_size) / n_of_patches < epoches:
			if (iter * batch_size) / n_of_patches > epoch_count:
				epoch_count = (iter * batch_size) / n_of_patches

			images, labels = dataset.next_batch(batch_size)
			sess.run([train_step], feed_dict={images_placeholder: images, labels_placeholder: labels})

			if epoch_count % 10:
				summary, ce, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={images_placeholder: images, labels_placeholder: labels})
				train_writer.add_summary(summary=summary, global_step=iter)
				print("training loss: " + str(ce))
				print("training acc: " + str(acc))

				total_val = 0
				val_iterations = int(validation_dataset.size() / batch_size)
				for i in range(val_iterations):
					val_images, val_labels = validation_dataset.next_batch(batch_size)
					summary, ce, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})
					total_val = total_val + acc
					test_writer.add_summary(summary=summary, global_step=iter)
				print("Val accuracy: " + str(float(total_val / val_iterations)))
			iter = iter + 1
if __name__ == '__main__':
	train(sys.argv[1:])