from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train = mnist.train.images  # [55000, 784]
labels = mnist.train.labels  # [55000, 10]

# Softmax Regressions
