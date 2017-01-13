import tensorflow as tf 
#Create a 1x2 input matrix
matrix1 = tf.constant([[3., 3.]])
#Create a 2x1 input matrix
matrix2 = tf.constant([[2.],[2.]])
#Store result in product using matrix multiplication op
product = tf.matmul(matrix1, matrix2)

#In order to run the operation we must create a session
sess = tf.Session()
result = sess.run(product)
print(result)
#Sessions should be closed to release resources
sess.close()

#Cleaner option using with block
#Automatically closes a session and releases resources when done
with tf.Session() as sess:
	result = sess.run([product])
	print(result)
