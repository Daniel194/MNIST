import tensorflow as tf
import numpy as np
import math
import Utility as util


class ConvolutionNeuralNetwork(object):
    def __init__(self, d, k):
        """
        :param d: dimensionality
        :param k: number of classes
        """

        # Parameter
        self.D = d
        self.K = k
        self.NR_VALIDATION_DATA = 50
        self.NR_ITERATION = 20000
        self.BATCH_SIZE = 50
        self.SHOW_ACC = 100

        # Hyperparameter
        self.TRAIN_STEP = 1e-4
        self.EPSILON = 1e-3
        self.BETA = 1e-3

        # Shape
        self.W1_1_SHAPE = [3, 3, 1, 128]
        self.W1_2_SHAPE = [3, 3, 128, 128]
        self.W2_1_SHAPE = [3, 3, 128, 256]
        self.W2_2_SHAPE = [3, 3, 256, 256]

        self.WFC_1_SHAPE = [12544, 4096]
        self.WFC_2_SHAPE = [4096, 2048]
        self.WFC_3_SHAPE = [2048, k]

        self.B1_SHAPE = [128]
        self.B2_SHAPE = [256]

        self.BFC1_SHAPE = [4096]
        self.BFC2_SHAPE = [2048]
        self.BFC3_SHAPE = [k]

    def training(self, features, labels):
        """
        Training the Convolutional Neural Network
        :param features: the training data [50000 x 784]
        :param labels: the true label for X  [50000 x 1]
        :return: return a dictionary which contains all learned parameters
        """

        # Preprocessing the data
        features = self.__preprocessing(features)  # [50000 x 784]

        # Split data into training and validation sets.
        train_features = features[self.NR_VALIDATION_DATA:]
        train_labels = labels[self.NR_VALIDATION_DATA:]
        validation_features = features[0:self.NR_VALIDATION_DATA]
        validation_labels = labels[0:self.NR_VALIDATION_DATA]

        # Launch the session
        sess = tf.InteractiveSession()

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.D])  # the data
        y_ = tf.placeholder(tf.float32, shape=[None, self.K])  # the true labels

        # Reshape
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # Initialize the weights and the biases
        # First Layer
        W1_1 = self.__weight_variable(self.W1_1_SHAPE)  # [ 3 x 3 x 1 x 128 ]
        scale1_1 = tf.Variable(tf.ones(self.B1_SHAPE))  # [128]
        beta1_1 = tf.Variable(tf.zeros(self.B1_SHAPE))  # [128]

        W1_2 = self.__weight_variable(self.W1_2_SHAPE)  # [ 3 x 3 x 128 x 128 ]
        scale1_2 = tf.Variable(tf.ones(self.B1_SHAPE))  # [128]
        beta1_2 = tf.Variable(tf.zeros(self.B1_SHAPE))  # [128]

        # Second Layer
        W2_1 = self.__weight_variable(self.W2_1_SHAPE)  # [ 3 x 3 x 128 x 256 ]
        scale2_1 = tf.Variable(tf.ones(self.B2_SHAPE))  # [256]
        beta2_1 = tf.Variable(tf.zeros(self.B2_SHAPE))  # [256]

        W2_2 = self.__weight_variable(self.W2_2_SHAPE)  # [ 3 x 3 x 256 x 256 ]
        scale2_2 = tf.Variable(tf.ones(self.B2_SHAPE))  # [256]
        beta2_2 = tf.Variable(tf.zeros(self.B2_SHAPE))  # [256]

        # Full Connected Layer 1
        WFC1 = self.__weight_variable(self.WFC_1_SHAPE)  # [ 12544 x 4096 ]
        scaleFC1 = tf.Variable(tf.ones(self.BFC1_SHAPE))  # [4096]
        betaFC1 = tf.Variable(tf.zeros(self.BFC1_SHAPE))  # [4096]

        # Full Connected Layer 2
        WFC2 = self.__weight_variable(self.WFC_2_SHAPE)  # [ 4096 x 2048 ]
        scaleFC2 = tf.Variable(tf.ones(self.BFC2_SHAPE))  # [2048]
        betaFC2 = tf.Variable(tf.zeros(self.BFC2_SHAPE))  # [2048]

        # Full Connected Layer 3
        WFC3 = self.__weight_variable(self.WFC_3_SHAPE)  # [ 2048 x 10 ]
        bFC3 = self.__bias_variable(self.BFC3_SHAPE)  # [10]

        # First Layer
        Z1_1 = self.__convolution(x_image, W1_1)
        batch_mean1_1, batch_var1_1 = tf.nn.moments(Z1_1, [0])
        BN1 = tf.nn.batch_normalization(Z1_1, batch_mean1_1, batch_var1_1, beta1_1, scale1_1, self.EPSILON)
        H1 = self.__activation(BN1)  # [50000 x 28 x 28 x 128]

        Z1_2 = self.__convolution(H1, W1_2)
        batch_mean1_2, batch_var1_2 = tf.nn.moments(Z1_2, [0])
        BN1_2 = tf.nn.batch_normalization(Z1_2, batch_mean1_2, batch_var1_2, beta1_2, scale1_2, self.EPSILON)
        H1_2 = self.__activation(BN1_2)  # [50000 x 28 x 28 x 128]

        H_pool1 = self.__pool(H1_2)  # [50000 x 14 x 14 x 128]

        # Second Layer
        Z2_1 = self.__convolution(H_pool1, W2_1)
        batch_mean2_1, batch_var2_1 = tf.nn.moments(Z2_1, [0])
        BN2 = tf.nn.batch_normalization(Z2_1, batch_mean2_1, batch_var2_1, beta2_1, scale2_1, self.EPSILON)
        H2 = self.__activation(BN2)  # [50000 x 14 x 14 x 256]

        Z2_2 = self.__convolution(H2, W2_2)
        batch_mean2_2, batch_var2_2 = tf.nn.moments(Z2_2, [0])
        BN2_2 = tf.nn.batch_normalization(Z2_2, batch_mean2_2, batch_var2_2, beta2_2, scale2_2, self.EPSILON)
        H2_2 = self.__activation(BN2_2)  # [50000 x 14 x 14 x 256]

        H_pool2 = self.__pool(H2_2)  # [50000 x 7 x 7 x 256]

        # First Full Connected Layer
        H_pool3_flat = tf.reshape(H_pool2, [-1, self.WFC_1_SHAPE[0]])  # [ 50000 x 12544 ]

        Z_FC1 = tf.matmul(H_pool3_flat, WFC1)
        batch_mean_fc1, batch_var_fc1 = tf.nn.moments(Z_FC1, [0])
        BN_FC1 = tf.nn.batch_normalization(Z_FC1, batch_mean_fc1, batch_var_fc1, betaFC1, scaleFC1, self.EPSILON)
        H_fc1 = self.__activation(BN_FC1)  # [ 50000 x 4096 ]

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        H_fc1_drop = tf.nn.dropout(H_fc1, keep_prob)  # [ 50000 x 4096 ]

        # Second Full Connected Layer
        Z_FC2 = tf.matmul(H_fc1_drop, WFC2)
        batch_mean_fc2, batch_var_fc2 = tf.nn.moments(Z_FC2, [0])
        BN_FC2 = tf.nn.batch_normalization(Z_FC2, batch_mean_fc2, batch_var_fc2, betaFC2, scaleFC2, self.EPSILON)
        H_fc2 = self.__activation(BN_FC2)  # [ 50000 x 2048 ]

        # Dropout
        H_fc2_drop = tf.nn.dropout(H_fc2, keep_prob)  # [ 50000 x 2048 ]

        # Third Full Connected Layer
        H_fc3 = tf.matmul(H_fc2_drop, WFC3) + bFC3  # [ 50000 x 10 ]

        y_conv = tf.nn.softmax(H_fc3)

        # Loss Function loss fn == avg(y'*log(y))
        loss_fn = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]) + self.BETA * tf.nn.l2_loss(
            W1_1) + self.BETA * tf.nn.l2_loss(W1_2) + self.BETA * tf.nn.l2_loss(W2_1) + self.BETA * tf.nn.l2_loss(
            W2_2) + self.BETA * tf.nn.l2_loss(WFC1) + self.BETA * tf.nn.l2_loss(WFC2) + self.BETA * tf.nn.l2_loss(WFC3)

        loss_fn_mean = tf.reduce_mean(loss_fn)

        # Training step - ADAM solver
        train_step = tf.train.AdamOptimizer(self.TRAIN_STEP).minimize(loss_fn_mean)

        # Evaluate the model
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        for i in range(self.NR_ITERATION):
            batch = util.generate_batch(train_features, train_labels, self.BATCH_SIZE)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % self.SHOW_ACC == 0:
                train_accuracy = accuracy.eval(feed_dict={x: validation_features, y_: validation_labels, keep_prob: 1})

                print('Step - ', i, ' - Acc : ', train_accuracy)

        W1_1_final = W1_1.eval()
        beta1_1_final = beta1_1.eval()
        scale1_1_final = scale1_1.eval()

        W1_2_final = W1_2.eval()
        beta1_2_final = beta1_2.eval()
        scale1_2_final = scale1_2.eval()

        W2_1_final = W2_1.eval()
        beta2_1_final = beta2_1.eval()
        scale2_1_final = scale2_1.eval()

        W2_2_final = W2_2.eval()
        beta2_2_final = beta2_2.eval()
        scale2_2_final = scale2_2.eval()

        WFC1_final = WFC1.eval()
        betaFC1_final = betaFC1.eval()
        scaleFC1_final = scaleFC1.eval()

        WFC2_final = WFC2.eval()
        betaFC2_final = betaFC2.eval()
        scaleFC2_final = scaleFC2.eval()

        WFC3_final = WFC3.eval()
        bFC3_final = bFC3.eval()

        # Close the session
        sess.close()

        return {
            'W1_1': W1_1_final,
            'beta1_1': beta1_1_final,
            'scale1_1': scale1_1_final,
            'W1_2': W1_2_final,
            'beta1_2': beta1_2_final,
            'scale1_2': scale1_2_final,
            'W2_1': W2_1_final,
            'beta2_1': beta2_1_final,
            'scale2_1': scale2_1_final,
            'W2_2': W2_2_final,
            'beta2_2': beta2_2_final,
            'scale2_2': scale2_2_final,
            'WFC1': WFC1_final,
            'betaFC1': betaFC1_final,
            'scaleFC1': scaleFC1_final,
            'WFC2': WFC2_final,
            'betaFC2': betaFC2_final,
            'scaleFC2': scaleFC2_final,
            'WFC3': WFC3_final,
            'bFC3': bFC3_final
        }

    def predict(self, test_features, nn):
        """
        Predict data
        :param test_features: testing data
        :param nn: it is a dictionary which contains a Neural Network
        :return: return the predicted labels and the accuracy
        """

        # Preprocessing
        test_features = self.__preprocessing(test_features)

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.D])  # the data

        W1_1 = tf.placeholder(tf.float32, shape=self.W1_1_SHAPE)  # the weights
        beta1_1 = tf.placeholder(tf.float32, shape=self.B1_SHAPE)  # the beta
        scale1_1 = tf.placeholder(tf.float32, shape=self.B1_SHAPE)  # the scale

        W1_2 = tf.placeholder(tf.float32, shape=self.W1_2_SHAPE)  # the weights
        beta1_2 = tf.placeholder(tf.float32, shape=self.B1_SHAPE)  # the beta
        scale1_2 = tf.placeholder(tf.float32, shape=self.B1_SHAPE)  # the scale

        W2_1 = tf.placeholder(tf.float32, shape=self.W2_1_SHAPE)  # the weights
        beta2_1 = tf.placeholder(tf.float32, shape=self.B2_SHAPE)  # the beta
        scale2_1 = tf.placeholder(tf.float32, shape=self.B2_SHAPE)  # the scale

        W2_2 = tf.placeholder(tf.float32, shape=self.W2_2_SHAPE)  # the weights
        beta2_2 = tf.placeholder(tf.float32, shape=self.B2_SHAPE)  # the beta
        scale2_2 = tf.placeholder(tf.float32, shape=self.B2_SHAPE)  # the scale

        WFC1 = tf.placeholder(tf.float32, shape=self.WFC_1_SHAPE)  # the weights
        betaFC1 = tf.placeholder(tf.float32, shape=self.BFC1_SHAPE)  # the beta
        scaleFC1 = tf.placeholder(tf.float32, shape=self.BFC1_SHAPE)  # the scale

        WFC2 = tf.placeholder(tf.float32, shape=self.WFC_2_SHAPE)  # the weights
        betaFC2 = tf.placeholder(tf.float32, shape=self.BFC2_SHAPE)  # the beta
        scaleFC2 = tf.placeholder(tf.float32, shape=self.BFC2_SHAPE)  # the scale

        WFC3 = tf.placeholder(tf.float32, shape=self.WFC_3_SHAPE)  # the weights
        bFC3 = tf.placeholder(tf.float32, shape=self.BFC3_SHAPE)  # the biases

        # Reshape
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First Layer
        Z1_1 = self.__convolution(x_image, W1_1)
        batch_mean1_1, batch_var1_1 = tf.nn.moments(Z1_1, [0])
        BN1 = tf.nn.batch_normalization(Z1_1, batch_mean1_1, batch_var1_1, beta1_1, scale1_1, self.EPSILON)
        H1 = self.__activation(BN1)  # [50000 x 28 x 28 x 128]

        Z1_2 = self.__convolution(H1, W1_2)
        batch_mean1_2, batch_var1_2 = tf.nn.moments(Z1_2, [0])
        BN1_2 = tf.nn.batch_normalization(Z1_2, batch_mean1_2, batch_var1_2, beta1_2, scale1_2, self.EPSILON)
        H1_2 = self.__activation(BN1_2)  # [50000 x 28 x 28 x 128]

        H_pool1 = self.__pool(H1_2)  # [50000 x 14 x 14 x 128]

        # Second Layer
        Z2_1 = self.__convolution(H_pool1, W2_1)
        batch_mean2_1, batch_var2_1 = tf.nn.moments(Z2_1, [0])
        BN2 = tf.nn.batch_normalization(Z2_1, batch_mean2_1, batch_var2_1, beta2_1, scale2_1, self.EPSILON)
        H2 = self.__activation(BN2)  # [50000 x 14 x 14 x 256]

        Z2_2 = self.__convolution(H2, W2_2)
        batch_mean2_2, batch_var2_2 = tf.nn.moments(Z2_2, [0])
        BN2_2 = tf.nn.batch_normalization(Z2_2, batch_mean2_2, batch_var2_2, beta2_2, scale2_2, self.EPSILON)
        H2_2 = self.__activation(BN2_2)  # [50000 x 14 x 14 x 256]

        H_pool2 = self.__pool(H2_2)  # [50000 x 7 x 7 x 256]

        # First Full Connected Layer
        H_pool3_flat = tf.reshape(H_pool2, [-1, self.WFC_1_SHAPE[0]])  # [ 50000 x 12544 ]

        Z_FC1 = tf.matmul(H_pool3_flat, WFC1)
        batch_mean_fc1, batch_var_fc1 = tf.nn.moments(Z_FC1, [0])
        BN_FC1 = tf.nn.batch_normalization(Z_FC1, batch_mean_fc1, batch_var_fc1, betaFC1, scaleFC1, self.EPSILON)
        H_fc1 = self.__activation(BN_FC1)  # [ 50000 x 4096 ]

        # Second Full Connected Layer
        Z_FC2 = tf.matmul(H_fc1, WFC2)
        batch_mean_fc2, batch_var_fc2 = tf.nn.moments(Z_FC2, [0])
        BN_FC2 = tf.nn.batch_normalization(Z_FC2, batch_mean_fc2, batch_var_fc2, betaFC2, scaleFC2, self.EPSILON)
        H_fc2 = self.__activation(BN_FC2)  # [ 50000 x 2048 ]

        # Third Full Connected Layer
        H_fc3 = tf.matmul(H_fc2, WFC3) + bFC3  # [ 50000 x 10]

        # Calculate the output
        y = tf.nn.softmax(H_fc3)  # [ 50000 x 10]

        # Launch the session
        sess = tf.InteractiveSession()

        # Initialize the placeholder
        feed_dict = {
            x: test_features,
            W1_1: nn['W1_1'],
            beta1_1: nn['beta1_1'],
            scale1_1: nn['scale1_1'],
            W1_2: nn['W1_2'],
            beta1_2: nn['beta1_2'],
            scale1_2: nn['scale1_2'],
            W2_1: nn['W2_1'],
            beta2_1: nn['beta2_1'],
            scale2_1: nn['scale2_1'],
            W2_2: nn['W2_2'],
            beta2_2: nn['beta2_2'],
            scale2_2: nn['scale2_2'],
            WFC1: nn['WFC1'],
            betaFC1: nn['betaFC1'],
            scaleFC1: nn['scaleFC1'],
            WFC2: nn['WFC2'],
            betaFC2: nn['betaFC2'],
            scaleFC2: nn['scaleFC2'],
            WFC3: nn['WFC3'],
            bFC3: nn['bFC3']
        }

        # Run model on test data
        predicted_labels = sess.run(y, feed_dict=feed_dict)

        # Close the session
        sess.close()

        # Convert SoftMax predictions to label
        predicted_labels = np.argmax(predicted_labels, axis=1)

        return predicted_labels

    def __weight_variable(self, shape):
        """
        Initialize the weights variable.
        :param shape: the shape.
        :return: return a TensorFlow variable
        """

        if len(shape) == 4:
            initial = np.random.randn(shape[0], shape[1], shape[2], shape[3]) * math.sqrt(
                2.0 / (shape[0] * shape[1] * shape[2] * shape[3]))
        else:
            initial = np.random.randn(shape[0], shape[1]) * math.sqrt(2.0 / (shape[0] * shape[1]))

        return tf.Variable(initial, dtype=tf.float32)

    def __preprocessing(self, X):
        """
        Preprocessing the X data by zero-centered and normalized them.
        :param X: the data.
        :return: return the new zero-centered and normalized data.
        """

        X = X.astype(np.float64)
        X = X - np.mean(X, dtype=np.float64)  # zero-centered
        X = X / np.std(X, dtype=np.float64)  # normalization

        return X

    def __bias_variable(self, shape):
        """
        Initialize the biases variable.
        :param shape:t he shape.
        :return: return a TensorFlow variable
        """

        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)

    def __convolution(self, x, W):
        """
        The convolution layer calculation.
        :param x: the data.
        :param W: the weights.
        :return: return the output of the convolution layer.
        """

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __activation(self, x):
        """
        The activation function.
        :param x: the data.
        :return: return the data after apply the activation.
        """

        return tf.nn.relu(x)

    def __pool(self, x):
        """
        The pool layer.
        :param x: the data.
        :return: return the output of the pool layer.
        """

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # Variable
    learn_data = 'RESULT_data/mnist'
    batch_size = 50
    features = util.read_features_from_csv('MNIST_data/train.csv')
    labels = util.read_labels_from_csv('MNIST_data/train.csv')
    test_features = util.read_features_from_csv('MNIST_data/test.csv', usecols=None)

    # Neural Network
    cnn = ConvolutionNeuralNetwork(784, 10)

    # Train the Neural Network
    if util.file_exist(learn_data):
        nn_parameter = util.unpickle(learn_data)
    else:
        nn_parameter = cnn.training(features, labels)

        util.pickle_nn(learn_data, nn_parameter)

    util.create_file('RESULT_data/submission_cnn2.csv')

    for i in range(0, test_features.shape[0], batch_size):
        batch_test_feature = test_features[i:i + batch_size, :]

        predicted_labels = cnn.predict(batch_test_feature, nn_parameter)

        # Save the predictions to label
        util.append_data_to_file('RESULT_data/submission_cnn2.csv', predicted_labels, i)
