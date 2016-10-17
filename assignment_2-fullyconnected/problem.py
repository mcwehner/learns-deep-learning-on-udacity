#!/usr/bin/env python3

import os

import numpy	as np
import tensorflow	as tf

from six.moves import cPickle as pickle
from six.moves import range

flags	= tf.app.flags
FLAGS	= flags.FLAGS

flags.DEFINE_float(	'learning_rate',	0.01,	'Initial learning rate.')
flags.DEFINE_integer(	'batch_size',	128,	'Batch size.')
flags.DEFINE_integer(	'hidden1',	1024,	'Number of units in hidden layer 1.')
flags.DEFINE_integer(	'max_steps',	3000,	'Number of steps to run the trainer.')
flags.DEFINE_string(	'train_dir',	'data',	'Directory to put the training data.')


IMAGE_SIZE	= 28
IMAGE_PIXELS	= IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS	= 10
PICKLE_FILE	= 'notMNIST.pickle'


# Reformat data:
# * Reshape data into flat matrix
# * Transform labels into 1-hot vectors
def reformat(dataset, labels):
	dataset	= dataset.reshape((-1, IMAGE_PIXELS)).astype(np.float32)

	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels	= (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)

	return dataset, labels

def placeholder_inputs():
	dataset_pl	= tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS))
	labels_pl	= tf.placeholder(tf.float32, shape=(FLAGS.batch_size, NUM_LABELS))

	return dataset_pl, labels_pl

def weight_variable(shape):
	initial	= tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name='weights')

def bias_variable(shape):
	# initial	= tf.constant(0.1, shape=shape)
	initial	= tf.zeros(shape)
	return tf.Variable(initial, name='bias')

def inference(images):
	with tf.name_scope('hidden1'):
		W_1	= weight_variable([IMAGE_PIXELS, FLAGS.hidden1])
		b_1	= bias_variable([FLAGS.hidden1])
		hidden1	= tf.nn.relu(tf.matmul(images, W_1) + b_1)

	with tf.name_scope('softmax_linear'):
		W_2	= weight_variable([FLAGS.hidden1, NUM_LABELS])
		b_2	= bias_variable([NUM_LABELS])
		logits	= tf.matmul(hidden1, W_2) + b_2

	return logits


def loss_calculation(logits, labels):
	cross_entropy	= tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
	loss	= tf.reduce_mean(cross_entropy, name='xentropy_mean')

	return loss

def training(loss, learning_rate):
	tf.scalar_summary(loss.op.name, loss)

	optimizer	= tf.train.GradientDescentOptimizer(learning_rate)
	global_step	= tf.Variable(0, name='global_step', trainable=False)
	train_op	= optimizer.minimize(loss, global_step=global_step)

	return train_op

def accuracy(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))

	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluation(logits):
	return tf.nn.softmax(logits)

def fill_feed_dict(step, dataset, labels, dataset_pl, labels_pl):
	# Pick an offset within the training data, which has been randomized.
	# Note: we could use better randomization across epochs.
	offset	= (step * FLAGS.batch_size) % (labels.shape[0] - FLAGS.batch_size)

	# Generate a minibatch.
	batch_data	= dataset[offset:(offset + FLAGS.batch_size), :]
	batch_labels	= labels[offset:(offset + FLAGS.batch_size), :]

	return { dataset_pl: batch_data, labels_pl: batch_labels }

# def do_eval(sess, eval_correct, dataset, labels, dataset_pl, labels_pl):
# 	mean_accuracy	= 0.0  # Counts the number of correct predictions.
# 	steps_per_epoch	= dataset.shape[0] // FLAGS.batch_size
# 	num_examples	= steps_per_epoch * FLAGS.batch_size
#
# 	for step in range(steps_per_epoch):
# 		feed_dict	= fill_feed_dict(step, dataset, labels, dataset_pl, labels_pl)
#
# 		mean_accuracy	+= accuracy(sess.run(eval_correct, feed_dict=feed_dict), feed_dict[labels_pl])
#
# 	mean_accuracy	= mean_accuracy / num_examples
#
# 	feed_dict	= fill_feed_dict(0, dataset, labels, dataset_pl, labels_pl)
#
# 	print('  Sample accuracy: %0.04f' % accuracy(sess.run(eval_correct, feed_dict=feed_dict)))
# 	print('  Mean accuracy: %0.04f' % mean_accuracy)

def do_eval(sess, eval_correct, dataset, labels, dataset_pl, labels_pl):
	# offset	= (np.random.randint(labels.shape[0]) * FLAGS.batch_size) % (labels.shape[0] - FLAGS.batch_size)

	feed_dict	= fill_feed_dict(np.random.randint(labels.shape[0]), dataset, labels, dataset_pl, labels_pl)
	predictions	= sess.run(eval_correct, feed_dict=feed_dict)

	print('  Accuracy: %f' % accuracy(predictions, feed_dict[labels_pl]))

def run_training():
	# Load previously pickled data
	with open(PICKLE_FILE, 'rb') as f:
		save = pickle.load(f)

		train_dataset	= save['train_dataset']
		train_labels	= save['train_labels']
		valid_dataset	= save['valid_dataset']
		valid_labels	= save['valid_labels']
		test_dataset	= save['test_dataset']
		test_labels	= save['test_labels']

		del save	# hint to help gc free up memory

	train_dataset, train_labels	= reformat(train_dataset, train_labels)
	valid_dataset, valid_labels	= reformat(valid_dataset, valid_labels)
	test_dataset, test_labels	= reformat(test_dataset, test_labels)

	print('Training set',	train_dataset.shape, train_labels.shape)
	print('Validation set',	valid_dataset.shape, valid_labels.shape)
	print('Test set',	test_dataset.shape, test_labels.shape)


	with tf.Graph().as_default():
		# Input data. For the training data, we use a placeholder that will be fed
		# at run time with a training minibatch.
		images_pl, labels_pl	= placeholder_inputs()

		logits	= inference(images_pl)
		loss	= loss_calculation(logits, labels_pl)
		train_op	= training(loss, FLAGS.learning_rate)

		train_eval	= evaluation(logits)
		valid_eval	= evaluation(logits)
		test_eval	= evaluation(logits)

		summary	= tf.merge_all_summaries()
		init	= tf.initialize_all_variables()
		saver	= tf.train.Saver()
		sess	= tf.Session()
		summary_writer	= tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		sess.run(init)

		print('Initialized')

		for step in range(FLAGS.max_steps):
			feed_dict	= fill_feed_dict(step, train_dataset, train_labels, images_pl, labels_pl)

			_, loss_value	= sess.run([train_op, loss], feed_dict=feed_dict)

			if 0 == step % 100:
				print('Minibatch loss at step %d: %f'	% (step, loss_value))
				# print('Minibatch eval:')
				# do_eval(sess, eval_correct, train_dataset, train_labels, images_pl, labels_pl)

				summary_str	= sess.run(summary, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if 0 == (step + 1) % 500 or FLAGS.max_steps == (step + 1):
				checkpoint_file	= os.path.join(FLAGS.train_dir, 'checkpoint')
				saver.save(sess, checkpoint_file, global_step=step)

				# print('Training data eval:')
				# do_eval(sess, eval_correct, train_dataset, train_labels, images_pl, labels_pl)

				print('Validation data eval:')
				do_eval(sess, valid_eval, valid_dataset, valid_labels, images_pl, labels_pl)

				print('Test data eval:')
				do_eval(sess, test_eval, test_dataset, test_labels, images_pl, labels_pl)

def main(_):
	run_training()

if '__main__' == __name__:
	tf.app.run()
