# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import data_handle

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import mlp


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 60000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 178, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def placeholder_inputs(batch_size, nfeatures):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    events_placeholder: events placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  events_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         nfeatures))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return events_placeholder, labels_placeholder


def fill_feed_dict(data_set, features_pl, labels_pl, keep_prob_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of 
    events_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  features_feed, labels_feed = data_set.getTrainBatch(FLAGS.batch_size)

  feed_dict = {
      features_pl: features_feed,
      labels_pl: labels_feed,
      keep_prob_pl: 0.7
  }
  return feed_dict


def do_eval(sess, eval_correct, features_pl, labels_pl,keep_prob_pl, data_set_iterator):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    events_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  num_examples = 0
  for features, labels in data_set_iterator:
    feed_dict = {
      features_pl: features,
      labels_pl: labels,
      keep_prob_pl: 1.0
    }
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    num_examples += len(labels) # just count how many examples we have seen
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = data_handle.DataHandle(0.2) 

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    events_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size, data_sets.Nfeatures)

    keep_prob_pl = tf.placeholder(tf.float32)

    # Build a Graph that computes predictions from the inference model.
    logits = mlp.inference(events_placeholder, keep_prob_pl,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mlp.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mlp.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mlp.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets,
                                 events_placeholder,
                                 labels_placeholder, keep_prob_pl)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                events_placeholder,
                labels_placeholder, keep_prob_pl,
                data_sets.iterateTrainData(FLAGS.batch_size))
        # Evaluate against the validation set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                events_placeholder,
                labels_placeholder, keep_prob_pl,
                data_sets.iterateTestData(FLAGS.batch_size))
        # Evaluate against the test set.


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
