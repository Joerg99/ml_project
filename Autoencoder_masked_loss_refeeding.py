import tensorflow as tf
import Data_handler
import matplotlib.pyplot as plt
import masking

n_input = n_output = 218
n_hidden = 210
n_hidden2 = 200
n_target_layer = 160

learning_rate = 0.01

X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
mask_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, n_input])

l2_reg = tf.contrib.layers.l2_regularizer(0.001)
he_init = tf.contrib.layers.variance_scaling_initializer()
def dense_layer(inputs, output, act = None, kernel_init = he_init, kernel_reg = l2_reg):
    return tf.layers.dense(inputs, output, activation=act, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)

layer_in = dense_layer(X, n_hidden)
#hidden_1 = dense_layer(layer_in, n_hidden)
target_layer = dense_layer(layer_in, n_target_layer)
target_dropout = tf.layers.dropout(target_layer)
#hidden_2 = dense_layer(target_layer, n_hidden)
layer_out = dense_layer(target_dropout, n_output)

########## tf.divide(tf.reduce_sum(tf.multiply(mask, tf.square(layer_out - X))), alle einsen in mask) 
########## mask = batch_size x n_input
loss_op_eval= tf.reduce_mean(tf.square(layer_out - X))
n_ones = n_input/2
loss_op = tf.divide(tf.reduce_sum(tf.multiply(mask_placeholder, tf.square(layer_out - X))),n_ones) # 3.0 = mask und 2 = anzahl von 1 in mask


optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)


#x_train = make_data(n_input, batch_size)
#x_test = make_data(n_input, batch_size)
all_data = Data_handler.generate_sparse_vectors()
batch_size= 400
data_train = all_data[:501]
print(len(data_train))
data_test = all_data[501:]
x_test = Data_handler.get_batch(data_test, 150)
epochs = 800

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses  = []
        eval_losses  = []
        for epoch in range(epochs):
            x_train = Data_handler.get_batch(data_train, batch_size)
            
            mask, n_ones = masking.bin_mask(x_train)
            _, loss= sess.run([train_op, loss_op], feed_dict={X:x_train, mask_placeholder:mask})
            
            ###### RE FEEDING
            #print(sess.run(layer_out, feed_dict= {X:x_train}))
            x_train = layer_out.eval(feed_dict={X:x_train})
            mask, n_ones = masking.bin_mask(x_train)
            _, loss= sess.run([train_op, loss_op], feed_dict={X:x_train, mask_placeholder:mask})
            
            losses.append(loss)
            print(loss)
            
            ####### EVAL
            eval_loss = sess.run(loss_op_eval, feed_dict={X:x_test})
            eval_losses.append(eval_loss)
            print(eval_loss, 'Epoch: ', epoch)
        results = layer_out.eval(feed_dict={X:x_test})
    
    
    for i in [1,4,6,8,15]:
        #print(x_test[i])
        results = [[int(x) for x in l] for l in results]
        j = 0
        for j in range(len(x_test[i])):
            if x_test[i][j] != 0:
                print(x_test[i][j], results[i][j])
        print(x_test[i])
        print(results[i])
            
    plt.plot(losses, 'r', label='train')
    plt.plot(eval_losses, 'g', label='eval')
    plt.legend(loc='upper right')
    plt.show()
