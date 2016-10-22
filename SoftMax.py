from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import the data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Launch the session
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])  # the data
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # the true labels

# Initialize the variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize all variables
sess.run(tf.initialize_all_variables())

# Calculate the output
y = tf.matmul(x, W) + b

# Apply cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
