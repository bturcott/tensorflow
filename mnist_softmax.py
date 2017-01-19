from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Import tensorflow library
import tensorflow as tf

#place holder value that will be used to run a computation
x = tf.placeholder(tf.float32, [None, 784])

#Weights and biases for model
#Variables are modifiable tensors that live in the graph of interacting
#operations. They can be used and modified by the computation
W = tf.Variable(tf.zeros([784, 10]))
b = tf. Variable(tf.zeros([10]))

#implementing the entire model...
y = tf.nn.softmax(tf.matmul(x, W) +b)

#Cross-entropy commonly used to determine the loss of a model
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Define traing step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize the created variables
init = tf.global_variables_initializer()

#Launch the model in a session and initialize the variables
sess = tf.Session()
sess.run(init)

#Run the training step 1000 times
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

#Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
