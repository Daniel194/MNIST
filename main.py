from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train = mnist.train.images  # [55000, 784]
labels = mnist.train.labels  # [55000, 10]

# Initialize weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax Regressions
x = tf.placeholder(tf.float32, [None, 784])  # [None, 784]

# Calculate the labels
y = tf.nn.softmax(tf.matmul(x, W) + b)


print(y)