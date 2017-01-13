import tensorflow as tf

#create a variable
state = tf.Variable(0, name="counter")

#Create a constant of one
one = tf.constant(1)
#add the two values together
new_value = tf.add(state, one)
#update the state with the new value
update = tf.assign(state, new_value)

#Variables must be initialized
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	#run the init_op 
	sess.run(init_op)
	#print the initial value of state
	print(sess.run(state))
	#run the op that updates 'state' and print 'state'
	#Updates the value of state and prints the result from 0 to 3
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
