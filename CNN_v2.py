import tensorflow as tf
import math
import time
import functools


class DigitsRecognition(object):
    def __init__(self):
        self.NUM_CLASSES = 10  # The MNIST dataset has 10 classes, representing the digits 0 through 9.

        # The MNIST images are always 28x28 pixels.
        self.IMAGE_SIZE = 28
        self.IMAGE_PIXELS = self.IMAGE_SIZE * self.IMAGE_SIZE

        self.learning_rate = 0.01  # Initial learning rate.
        self.max_steps = 20000  # Number of steps to run trainer.
        self.batch_size = 100  # Batch size.  Must divide evenly into the dataset sizes.

        self.W_conv1_shape = [3, 3, 1, 32]
        self.b_conv1_shape = [32]

        self.W_conv2_shape = [3, 3, 1, 32]
        self.b_conv2_shape = [32]

        self.W_conv3_shape = [3, 3, 32, 64]
        self.b_conv3_shape = [64]

        self.W_conv4_shape = [3, 3, 32, 64]
        self.b_conv4_shape = [64]

        self.W_fc1_shape = [7 * 7 * 64, 1024]
        self.b_fc1_shape = [1024]

        self.W_fc2_shape = [1024, 10]
        self.b_fc2_shape = [10]

    def training(self, training, validation, test):
        """
        Train MNIST for a number of steps.
        :param training:
        :param validation:
        :param test:
        :return:
        """

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the images and labels.
            images_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.IMAGE_PIXELS))
            labels_placeholder = tf.placeholder(tf.int32, shape=self.batch_size)

            # Build a Graph that computes predictions from the inference model.
            logits = self.__inference(images_placeholder)

            # Add to the Graph the Ops for loss calculation.
            loss = self.__loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.__training(loss)

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = self.__evaluation(logits, labels_placeholder)

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(tf.initialize_all_variables())

            # Start the training loop.
            for step in range(self.max_steps):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                feed_dict = self.__fill_feed_dict(training, images_placeholder, labels_placeholder)

                # Run one step of the model.  The return values are the activations from the
                # `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == self.max_steps:
                    # Evaluate against the training set.
                    print('Training Data Eval:')

                    self.__do_eval(sess, eval_correct, images_placeholder, labels_placeholder, training)

                    # Evaluate against the validation set.
                    print('Validation Data Eval:')

                    self.__do_eval(sess, eval_correct, images_placeholder, labels_placeholder, validation)

                    # Evaluate against the test set.
                    print('Test Data Eval:')

                    self.__do_eval(sess, eval_correct, images_placeholder, labels_placeholder, test)

    def __do_eval(self, sess, eval_correct, images_placeholder, labels_placeholder, data_set):
        """
        Runs one evaluation against the full epoch of data.
        :param sess: The session in which the model has been trained.
        :param eval_correct: The Tensor that returns the number of correct predictions.
        :param images_placeholder: The images placeholder.
        :param labels_placeholder: The labels placeholder.
        :param data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
        """

        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size

        for step in range(steps_per_epoch):
            feed_dict = self.__fill_feed_dict(data_set, images_placeholder, labels_placeholder)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = true_count / num_examples

        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

    def __fill_feed_dict(self, data_set, images_pl, labels_pl):
        """
        Fills the feed_dict for training the given step.
        :param data_set: The set of images and labels, from input_data.read_data_sets()
        :param images_pl: he images placeholder, from placeholder_inputs().
        :param labels_pl: The labels placeholder, from placeholder_inputs().
        :return: The feed dictionary mapping from placeholders to values.
        """

        # Create the feed_dict for the placeholders filled with the next `batch size` examples.
        images_feed, labels_feed = data_set.next_batch(self.batch_size)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
        }

        return feed_dict

    def __inference(self, images):
        """
        Build the MNIST model up to where it may be used for inference.
        :param images: Images placeholder, from inputs().
        :return: Output tensor with the computed logits.
        """

        # First Convolutional Layer
        with tf.name_scope('hidden1'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv1_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv1_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_conv1_shape), name='biases')
            hidden1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

        # Second Convolutional Layer
        with tf.name_scope('hidden2'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv2_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv2_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_conv2_shape), name='biases')
            hidden2 = tf.nn.relu(tf.nn.conv2d(hidden1, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

        # First Pool Layer
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Third Convolutional Layer
        with tf.name_scope('hidden3'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv3_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv3_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_conv3_shape), name='biases')
            hidden3 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

        # Fourth Convolutional Layer
        with tf.name_scope('hidden4'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv4_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv4_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_conv4_shape), name='biases')
            hidden4 = tf.nn.relu(tf.nn.conv2d(hidden3, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)

        # Second Pool Layer
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(hidden4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First Fully Connected Layer
        with tf.name_scope('fc1'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_fc1_shape)
            pool2_flat = tf.reshape(pool2, [-1, self.W_fc1_shape[0]])

            weights = tf.Variable(tf.truncated_normal(self.W_fc1_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_fc1_shape), name='biases')
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

        # First Dropout
        with tf.name_scope('dropout1'):
            keep_prob = tf.placeholder(tf.float32)
            dropout1 = tf.nn.dropout(fc1, keep_prob)

        # Second Fully Connected Layer
        with tf.name_scope('fc2'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_fc2_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_fc2_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_fc2_shape), name='biases')
            fc2 = tf.matmul(dropout1, weights) + biases

        return fc2

    @staticmethod
    def __loss(logits, labels):
        """
        Calculates the loss from the logits and the labels.
        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param labels: Labels tensor, int32 - [batch_size].
        :return: Loss tensor of type float.
        """

        labels = tf.to_int64(labels)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        return loss

    def __training(self, loss):
        """
        Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        :param loss: Loss tensor, from loss().
        :return: The Op for training.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    @staticmethod
    def __evaluation(logits, labels):
        """
        Evaluate the quality of the logits at predicting the label.
        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
        :return: A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
        """

        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)

        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))


def main(_):
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)

    tf.gfile.MakeDirs(log_dir)

    # run_training()


if __name__ == '__main__':
    # Directory to put the input data.
    log_dir = '/tmp/tensorflow/mnist/input_data'

    # Directory to put the input data.
    input_data_dir = '/tmp/tensorflow/mnist/input_data'

    tf.app.run(main=main)
