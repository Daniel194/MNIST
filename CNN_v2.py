import tensorflow as tf
import math


class DigitsRecognition(object):
    def __init__(self):
        # The MNIST dataset has 10 classes, representing the digits 0 through 9.
        self.NUM_CLASSES = 10

        # The MNIST images are always 28x28 pixels.
        self.IMAGE_SIZE = 28
        self.IMAGE_PIXELS = self.IMAGE_SIZE * self.IMAGE_SIZE

    def __inference(self, images, hidden1_units, hidden2_units):
        """
        Build the MNIST model up to where it may be used for inference.
        :param images: Images placeholder, from inputs().
        :param hidden1_units: Size of the first hidden layer.
        :param hidden2_units: Size of the second hidden layer.
        :return: Output tensor with the computed logits.
        """

        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([self.IMAGE_PIXELS, hidden1_units],
                                                      stddev=1.0 / math.sqrt(float(self.IMAGE_PIXELS))), name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                      stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(tf.truncated_normal([hidden2_units, self.NUM_CLASSES],
                                                      stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
            biases = tf.Variable(tf.zeros([self.NUM_CLASSES]), name='biases')
            logits = tf.matmul(hidden2, weights) + biases

        return logits

    def __loss(self, logits, labels):
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

    def __training(self, loss, learning_rate):
        """
        Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        :param loss: Loss tensor, from loss().
        :param learning_rate: The learning rate to use for gradient descent.
        :return: The Op for training.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def __evaluation(self, logits, labels):
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
