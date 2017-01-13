import tensorflow as tf 

#constant ops to generate input data
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
#add and multiple opps to manipulate input data
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)


#run the operation and print the result
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print(result)
