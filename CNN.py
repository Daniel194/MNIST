import tensorflow as tf
import numpy as np
import Utility as util


class DigitsRecognition(object):
    def training(self, features, labels):
        # Split data into training and validation sets.
        train_features = features[50:]
        train_labels = labels[50:]
        validation_features = features[0:50]
        validation_labels = labels[0:50]

        # Launch the session
        self.sess = tf.InteractiveSession()

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])  # the data
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # the true labels

        # Prepare the data
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # First Layer
        self.W_conv1 = self.__weight_variable([3, 3, 1, 32])
        self.b_conv1 = self.__bias_variable([32])

        self.h_conv1 = tf.nn.relu(self.__conv(self.x_image, self.W_conv1) + self.b_conv1)

        self.W_conv1_2 = self.__weight_variable([3, 3, 32, 32])
        self.b_conv1_2 = self.__bias_variable([32])

        self.h_conv1_2 = tf.nn.relu(self.__conv(self.h_conv1, self.W_conv1_2) + self.b_conv1_2)
        self.h_pool1 = self.__max_pool(self.h_conv1_2)

        # Second Layer
        self.W_conv2 = self.__weight_variable([3, 3, 32, 64])
        self.b_conv2 = self.__bias_variable([64])

        self.h_conv2 = tf.nn.relu(self.__conv(self.h_pool1, self.W_conv2) + self.b_conv2)

        self.W_conv2_2 = self.__weight_variable([3, 3, 64, 64])
        self.b_conv2_2 = self.__bias_variable([64])

        self.h_conv2_2 = tf.nn.relu(self.__conv(self.h_conv2, self.W_conv2_2) + self.b_conv2_2)
        self.h_pool2 = self.__max_pool(self.h_conv2_2)

        # First Full Connected Layer
        self.W_fc1 = self.__weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.__bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # Second Full Connected Layer
        self.W_fc2 = self.__weight_variable([1024, 10])
        self.b_fc2 = self.__bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        # Loss Function loss fn == avg(y'*log(y))
        loss_fn = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

        # Training step - ADAM solver
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_fn)

        # Evaluate the model
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize all variables
        self.sess.run(tf.initialize_all_variables())

        for i in range(20000):
            batch = util.generate_batch(train_features, train_labels, 50)
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.75})

            if i % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={self.x: validation_features, self.y_: validation_labels, self.keep_prob: 1.0})

                print('Step - ', i, ' - Acc : ', train_accuracy)

    def predicting(self, test_features):
        # Run model on test data
        predicted_labels = self.sess.run(self.y_conv, feed_dict={self.x: test_features, self.keep_prob: 1.0})

        # Convert softmax predictions to label
        predicted_labels = np.argmax(predicted_labels, axis=1)

        return predicted_labels

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __conv(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # Read the feature and the labels.
    features = util.read_features_from_csv('MNIST_data/train.csv')
    labels = util.read_labels_from_csv('MNIST_data/train.csv')
    test_features = util.read_features_from_csv('MNIST_data/test.csv', usecols=None)

    model = DigitsRecognition()

    model.training(features, labels)

    predicted_labels = model.predicting(test_features)

    # Save the predictions to label
    util.save_predicted_labels('RESULT_data/submission_cnn.csv', predicted_labels)

    # Close the session
    model.sess.close()
