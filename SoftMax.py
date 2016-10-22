import tensorflow as tf
import numpy as np
import Utility as util

# Read the feature and the labels.
features = util.read_features_from_csv('MNIST_data/train.csv')
labels = util.read_labels_from_csv('MNIST_data/train.csv')
test_features = util.read_features_from_csv('MNIST_data/test.csv', usecols=None)

# Split data into training and validation sets.
train_features = features[800:]
train_labels = labels[800:]
validation_features = features[0:800]
validation_labels = labels[0:800]

# Launch the session
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])  # the data
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # the true labels

# Initialize the variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Calculate the output
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Apply cross entropy loss fn == avg(y'*log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = util.generate_batch(train_features, train_labels, 100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    if i % 100 == 0:
        print('Step - ', i, ' - Acc : ', accuracy.eval(feed_dict={x: validation_features, y_: validation_labels}))

# Run model on test data
predicted_labels = sess.run(y, feed_dict={x: test_features})

# Convert softmax predictions to label
predicted_labels = np.argmax(predicted_labels, axis=1)

# Save the predictions to label
util.save_predicted_labels('RESULT_data/submission.csv', predicted_labels)

# Close the session
sess.close()
