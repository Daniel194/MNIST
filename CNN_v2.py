import tensorflow as tf
import math
import time
import functools
import Utility
import numpy as np
import sys


class DigitsRecognition(object):
    def __init__(self):

        self.learning_rate = 1e-4  # Initial learning rate.
        self.max_steps = 20000  # Number of steps to run trainer.
        self.batch_size = 100  # Batch size.  Must divide evenly into the dataset sizes.

        self.IMAGE_SIZE = 28  # The size of the image in weight and height.
        self.NR_CHANEL = 1  # The number of chanel.
        self.IMAGE_SHAPE = (self.batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NR_CHANEL)

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

        self.dropout_probability = 0.75

    def prediction(self, training, training_labels, validation, validation_labels, test):
        """
        Train MNIST for a number of steps.
        :param training: The training future.
        :param training_labels: The true labels for the training future.
        :param validation: The validation data.
        :param validation_labels: The true labels for validation labels.
        :param test: The test data.
        :return: Return all labels of test data.
        """

        # Preprocessing the training, validation adn test data.
        training = self.__data_preprocessing(training)
        validation = self.__data_preprocessing(validation)
        test = self.__data_preprocessing(test)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the images, labels and dropout probability.
            images_placeholder = tf.placeholder(tf.float32, shape=self.IMAGE_SHAPE)
            labels_placeholder = tf.placeholder(tf.int32, shape=self.batch_size)
            keep_prob = tf.placeholder(tf.float32)

            # Build a Graph that computes predictions from the inference model.
            logits = self.__inference(images_placeholder, keep_prob)

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
                images, images_labels = self.__generate_batch(training, training_labels)
                feed_dict = {images_placeholder: images,
                             labels_placeholder: images_labels,
                             keep_prob: self.dropout_probability}

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
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')

                    self.__do_eval(sess, eval_correct, validation, validation_labels, images_placeholder,
                                   labels_placeholder, keep_prob)

            return self.__prediction(sess, logits, test, images_placeholder, keep_prob)

    def __do_eval(self, sess, eval_correct, data, data_labels, images_placeholder, labels_placeholder, keep_prob):
        """
        Runs one evaluation against the full epoch of data.
        :param sess: The session in which the model has been trained.
        :param eval_correct: The Tensor that returns the number of correct predictions.
        :param data: The validation data.
        :param data_labels: The true label of validation data.
        :param images_placeholder: The images placeholder.
        :param labels_placeholder: The labels placeholder.
        :param keep_prob: The probability to keep a neurone active.
        """

        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data.shape[0] // self.batch_size
        num_examples = steps_per_epoch * self.batch_size

        for step in range(0, num_examples, self.batch_size):
            validation_batch = data[step:step + self.batch_size, :]
            validation_batch_labels = data_labels[step:step + self.batch_size, :]

            feed_dict = {images_placeholder: validation_batch,
                         labels_placeholder: validation_batch_labels,
                         keep_prob: 1.0}
            true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = true_count / num_examples

        print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

    def __inference(self, images, keep_prob):
        """
        Build the MNIST model up to where it may be used for inference.
        :param images: Images placeholder, from inputs().
        :param keep_prob: the probability to keep a neuron data in Dropout Layer.
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
    def __loss(softmax_logits, true_labels):
        """
        Calculates the loss from the logits and the labels.
        :param softmax_logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param true_labels: Labels tensor, int32 - [batch_size].
        :return: Loss tensor of type float.
        """

        true_labels = tf.to_int64(true_labels)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(softmax_logits, true_labels, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        return loss

    def __training(self, loss):
        """
        Sets up the training Ops.
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
    def __evaluation(logits, true_labels):
        """
        Evaluate the quality of the logits at predicting the label.
        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param true_labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
        :return: A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
        """

        # Top k correct prediction
        correct = tf.nn.in_top_k(logits, true_labels, 1)

        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def __generate_batch(self, data, data_labels):
        """
        Generate Batches.
        :param data: the features data.
        :param data_labels: the labels data.
        :return: return labels and features.
        """

        batch_indexes = np.random.random_integers(0, len(data) - 1, self.batch_size)
        batch_dat = data[batch_indexes]
        batch_labels = data_labels[batch_indexes]

        return batch_dat, batch_labels

    def __data_preprocessing(self, data):
        """
        Preprocesing the MNIST data.
        :param data: the data.
        :return: the zero-centered and normalization data.
        """

        data -= np.mean(data, dtype=np.float64)  # zero-centered
        data /= np.std(data, dtype=np.float64)  # normalization

        return tf.reshape(data, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NR_CHANEL])

    def __prediction(self, sess, logits, data, images_placeholder, keep_prob):
        """
        Predicting the labels of the data.
        :param sess: The session in which the model has been trained.
        :param logits: The tenssor that calculate the logits.
        :param data: The data.
        :param images_placeholder: The images placeholder.
        :param keep_prob: The probability to keep a neurone active.
        :return: return the labels predicted.
        """

        steps_per_epoch = data.shape[0] // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        predicted_labels = []

        for step in range(0, num_examples, self.batch_size):
            data_batch = data[step:step + self.batch_size, :]
            feed_dict = {images_placeholder: data_batch,
                         keep_prob: 1.0}

            # Run model on test data
            batch_predicted_labels = sess.run(logits, feed_dict=feed_dict)
            batch_predicted_labels = tf.nn.softmax(batch_predicted_labels)

            # Convert softmax predictions to label and append to all results.
            batch_predicted_labels = np.argmax(batch_predicted_labels, axis=1)
            predicted_labels.append(batch_predicted_labels)

        sess.close()

        return predicted_labels


if __name__ == '__main__':
    # CONSTANTS
    TRAIN_DATA = 'MNIST_data/train.csv'
    TEST_DATA = 'MNIST_data/test.csv'
    SAVE_DATA = 'RESULT_data/submission_cnn_v2.csv'
    OUTPUT_FILE = 'RESULT_data/output_cnn_v2.txt'

    # Redirect the output to a file
    sys.stdout = open(OUTPUT_FILE, 'w')

    # Read the feature and the labels.
    features = Utility.read_features_from_csv(TRAIN_DATA)
    labels = Utility.read_labels_from_csv(TRAIN_DATA)
    test_features = Utility.read_features_from_csv(TEST_DATA, usecols=None)

    train_features = features[5000:]
    train_labels = labels[5000:]
    validation_features = features[0:5000]
    validation_features_labels = labels[0:5000]

    model = DigitsRecognition()

    predictions = model.prediction(train_features, train_labels, validation_features,
                                   validation_features_labels, test_features)

    # Save the predictions to label
    Utility.create_file('RESULT_data/submission_cnn_v2.csv')

    Utility.write_to_file(SAVE_DATA, predictions)
