import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


n_inputs = n_outputs= 28*28

n_target = 20

learning_rate= 0.01


def dense_layer(input_data, number_neurons, activation):
    
    return tf.layers.dense(input_data, number_neurons, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

X = tf.placeholder(tf.float32, [None, n_inputs])

target_mean = dense_layer(X, n_target, None)
target_gamma = dense_layer(X, n_target, None)
noise = tf.random_normal(tf.shape(target_gamma), dtype=tf.float32)
target_layer = dense_layer(X, n_target, tf.nn.relu)

logits = dense_layer(target_layer, n_outputs, None)
output_layer = tf.sigmoid(logits)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
recon_loss = tf.reduce_sum(cross_entropy)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(target_gamma) + tf.square(target_mean) - 1 - target_gamma)

loss_op = recon_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss_op)

n_digits = 10
epochs = 1
batch_size = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        print(epoch)
        n_batches = mnist.train.num_examples
        for it in range(n_batches):
            print(it)
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss = sess.run([training_op, loss_op], feed_dict={X:x_batch})
    
        codings_rnd = np.random.normal(size=[n_digits, n_target])
        outputs_val = output_layer.eval(feed_dict={target_layer:codings_rnd})


plt.figure(figsize=(8,50))
for it in range(n_digits):
    plt.subplot(n_digits, 10, it + 1)
    plot_image(outputs_val[it])
plt.show()
    




