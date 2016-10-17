#!/usr/bin/env python3

import collections
import math
import os
import pickle
import time

import numpy	as np
import tensorflow	as tf


IMAGE_SIZE	= 28
IMAGE_PIXELS	= IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS	= 10

flags	= tf.app.flags
FLAGS	= flags.FLAGS

flags.DEFINE_float(	'learning_rate',	0.5,	'Initial learning rate.')
flags.DEFINE_integer(	'batch_size',	128,	'Batch size.')
flags.DEFINE_integer(	'max_steps',	5000,	'Number of steps to run the trainer.')
flags.DEFINE_string(	'train_dir',	'data',	'Directory to put the training data.')


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class Dataset(object):
	def __init__(self, images, labels):
		self._images	= images.reshape((-1, IMAGE_PIXELS)).astype(np.float32)
		self._labels	= (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
		self._num_examples	= self._images.shape[0]
		self._index_in_epoch	= 0
		self._epochs_completed	= 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	def next_batch(self, batch_size):
		start	= self._index_in_epoch
		self._index_in_epoch	+= batch_size

		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1

			# Shuffle the data
			permutation = np.arange(self._num_examples)

			np.random.shuffle(permutation)
			self._images = self._images[permutation]
			self._labels = self._labels[permutation]

			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size

			assert batch_size <= self._num_examples

		end = self._index_in_epoch

		return self._images[start:end], self._labels[start:end]

def load_pickled_data(filename='notMNIST.pickle'):
	with open(filename, 'rb') as fh:
		save = pickle.load(fh)

		train	= Dataset(save['train_dataset'], save['train_labels'])
		validation	= Dataset(save['valid_dataset'], save['valid_labels'])
		test	= Dataset(save['test_dataset'], save['test_labels'])

		del save	# hint to help gc free up memory

		return Datasets(train=train, validation=validation, test=test)

def weight_variable(shape):
	return tf.Variable(
		# tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
		tf.truncated_normal(shape, stddev=0.1),
		name='weight',
	)

def bias_variable(shape):
	return tf.Variable(tf.zeros(shape), name='bias')

def nn_layer(name, input_tensor, input_size, output_size, act=tf.nn.relu):
	with tf.name_scope(name):
		with tf.name_scope('weights'):
			weights	= weight_variable([input_size, output_size])
			variable_summaries(name + '/weights', weights)

		with tf.name_scope('bias'):
			bias	= bias_variable([output_size])
			variable_summaries(name + '/bias', bias)

		with tf.name_scope('preactivate'):
			preactivate	= tf.matmul(input_tensor, weights) + bias
			tf.histogram_summary(name + '/preactivations', preactivate)

		activations	= act(preactivate, name='activation')
		tf.histogram_summary(name + '/activations', activations)

		return activations


def variable_summaries(name, var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)

		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

		tf.scalar_summary('stddev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

def model_inference(images, *hidden_units):
	last_layer	= images
	last_size	= IMAGE_PIXELS

	for i, hidden_size in enumerate(hidden_units):
		last_layer	= nn_layer('hidden_%d' % i, last_layer, last_size, hidden_size)
		last_size	= hidden_size

	logits	= nn_layer('softmax', last_layer, last_size, NUM_LABELS, act=tf.nn.softmax)

	return logits

def model_loss(logits, labels):
	cross_entropy	= tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
	loss	= tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

	return loss

def model_training(loss, learning_rate):
	tf.scalar_summary(loss.op.name, loss)

	optimizer	= tf.train.GradientDescentOptimizer(learning_rate)
	global_step	= tf.Variable(0, name='global_step', trainable=False)
	train_op	= optimizer.minimize(loss, global_step=global_step)

	return train_op

def model_evaluation(logits, labels):
	with tf.name_scope('evaluation'):
		correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

		return tf.reduce_sum(tf.cast(correct, tf.float32))

def fill_feed_dict(dataset, images, labels):
	images_feed, labels_feed	= dataset.next_batch(FLAGS.batch_size)

	return {
		images:	images_feed,
		labels:	labels_feed,
	}

def do_eval(session, eval_correct, images, labels, dataset):
	true_count	= 0
	steps_per_epoch	= dataset.num_examples // FLAGS.batch_size
	num_examples	= steps_per_epoch * FLAGS.batch_size

	for step in range(steps_per_epoch):
		feed_dict	= fill_feed_dict(dataset, images, labels)
		true_count	+= session.run(eval_correct, feed_dict=feed_dict)

	precision	= true_count / num_examples

	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def main(_):
	datasets	= load_pickled_data()

	with tf.Graph().as_default():
		# Placeholders
		images	= tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS), name='images')
		labels	= tf.placeholder(tf.float32, shape=(FLAGS.batch_size, NUM_LABELS), name='labels')

		# Main ops
		logits	= model_inference(images, 1024, 512)
		loss	= model_loss(logits, labels)
		train_op	= model_training(loss, FLAGS.learning_rate)
		eval_correct	= model_evaluation(logits, labels)

		# Setup
		summary	= tf.merge_all_summaries()
		init	= tf.initialize_all_variables()
		saver	= tf.train.Saver()
		session	= tf.Session()
		summary_writer	= tf.train.SummaryWriter(FLAGS.train_dir, session.graph)

		session.run(init)

		for step in range(FLAGS.max_steps):
			start_time	= time.time()

			feed_dict	= fill_feed_dict(datasets.train, images, labels)
			_, loss_value	= session.run([train_op, loss], feed_dict=feed_dict)

			duration	= time.time() - start_time

			if 0 == step % 100:
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

				summary_str	= session.run(summary, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if 0 == (1 + step) % 1000 or FLAGS.max_steps == (1 + step):
				checkpoint_file	= os.path.join(FLAGS.train_dir, 'checkpoint')
				saver.save(session, checkpoint_file, global_step=step)

				# print('Training data eval:')
				# do_eval(session, eval_correct, images, labels, datasets.train)

				print('Validation data eval:')
				do_eval(session, eval_correct, images, labels, datasets.validation)

				print('Test data eval:')
				do_eval(session, eval_correct, images, labels, datasets.test)


if '__main__' == __name__:
	tf.app.run()
