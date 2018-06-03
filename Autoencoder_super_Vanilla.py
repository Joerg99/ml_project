import tensorflow as tf
import numpy as np
import Data_handler

def make_data(seq_length, batch_size):
    result = np.zeros((batch_size, seq_length))
        
    for i in range(len(result)):
        result[i] = np.random.randint(4 , size=seq_length)#list(range(seq_length)) #/2)) + list(reversed(range((seq_length/2) )))

    return result




n_input = n_output = 9124
n_hidden = 9000

n_target_layer = 6000

learning_rate = 0.01

X = tf.placeholder(tf.float16, [None, n_input])

hidden_layer_in = tf.layers.dense(X, n_hidden)
target_layer = tf.layers.dense(hidden_layer_in, n_target_layer)
hidden_layer_out = tf.layers.dense(target_layer, n_output)
#output_layer = tf.layers.dense(hidden_layer_in, n_output)

loss_op= tf.reduce_mean(tf.square(hidden_layer_out - X))

optimizer = tf.train.AdamOptimizer(learning_rate)

train_op = optimizer.minimize(loss_op)


batch_size= 2
#x_train = make_data(n_input, batch_size)
#x_test = make_data(n_input, batch_size)
data = Data_handler.generate_dense_vectors()
x_train = Data_handler.get_batch(data, batch_size)
x_test = Data_handler.get_batch(data, batch_size)
epochs = 50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses  = []
    for _ in range(epochs):
        _, loss= sess.run([train_op, loss_op], feed_dict={X:x_train})
        print(loss)
        losses.append(loss)
    results = hidden_layer_out.eval(feed_dict={X:x_test})
    low_dim_rep = target_layer.eval(feed_dict={X:x_test})
    
for i in range(len(x_test)):
    print(x_test[i] ,'    ', results[i])

print(low_dim_rep)
    