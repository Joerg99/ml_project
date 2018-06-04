import tensorflow as tf

#bin = [[0,1,0,1],[1,1,0,1],[1,0,0,1],[0,0,1,0],[0,0,0,1]]
batch_size = 5
n_input = 4
target = tf.Variable([[0,10,0,3],[3,5,0,7],[3,0,0,2],[0,0,2,0],[0,0,0,5]])
y = tf.constant([[2,20,44,5],[2,5,2,7],[0,1,0,9],[1,2,1,0],[1,0,0,8]])
#y = tf.transpose(y)
#op = tf.matmul(target, y)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #target_var = tf.Variable(target)
    #y_var = tf.Variable(y)
    
    mask = sess.run(tf.equal(target, 0))
    print(mask)
    
    masked_y = sess.run(tf.boolean_mask(y,mask))
    print(len(masked_y))
    print(sess.run(tf.multiply(masked_y,masked_y)))
    
    
    #sess.run(target.assign(y))
    #res = sess.run(op)
    #y[0][0] = 666
    #print(sess.run(tf.divide(tf.reduce_sum(tf.square(tf.multiply(target, y))),2)))
    #print(res)
    
    
    
'''
def bin_mask(lol, batch_size, n_input):
    bin_lol = []
     
    for i in range(batch_size):
        bin_l = [0] * n_input
        for j in range(n_input):
            print(lol[i][j])
            if lol[i][j] != 0:
                bin_l[j] = 1
 
        bin_lol.append(bin_l)
    return bin_lol

blah = bin_mask(y, batch_size, n_input)
print(blah)
'''
# for i in range(len(b)):
#     for j in range(len(b[0])):
#         print(( blah[i][j] * (b[i][j] - target[i][j])**2) / len(b[0]))

