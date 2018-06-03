import tensorflow as tf
import numpy as np
import Data_handler






n_input = n_output = 1303
n_hidden = 1300

n_target_layer = 1200

learning_rate = 0.01

X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])



layer_in = tf.layers.dense(X, n_input)
hidden_1  = tf.layers.dense(layer_in, n_hidden)
target_layer = tf.layers.dense(layer_in, n_target_layer)
hidden_2 = tf.layers.dense(target_layer, n_hidden)
layer_out = tf.layers.dense(target_layer, n_output)

loss_op= tf.reduce_mean(tf.square(layer_out - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)


#x_train = make_data(n_input, batch_size)
#x_test = make_data(n_input, batch_size)
data = Data_handler.generate_sparse_vectors()
batch_size= len(data)
x_train = Data_handler.get_batch(data, batch_size)
x_test = Data_handler.get_batch(data, batch_size)
epochs = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses  = []
    for epoch in range(epochs):
        _, loss= sess.run([train_op, loss_op], feed_dict={X:x_train})
        print(loss, 'Epoch: ', epoch)
        #print(x_train[1])
        losses.append(loss)
    results = layer_out.eval(feed_dict={X:x_test})
        #if epoch % 100 == 0:
        #    x_train = results
    #low_dim_rep = target_layer.eval(feed_dict={X:x_test})


for i in [1,4,6,8,15]:
    #print(x_test[i])
    results = [[int(x) for x in l] for l in results]
    j = 0
    for j in range(len(x_test[i])):
        if x_test[i][j] != 0:
            print(x_test[i][j], results[i][j])
    print(x_test[i])
    print(results[i])
        


#print(low_dim_rep)
    