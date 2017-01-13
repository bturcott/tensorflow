import tensorflow as tf 

#use placeholder to create feed operations
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
#Feeds only used for the run call
with tf.Session() as sess:
	print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
