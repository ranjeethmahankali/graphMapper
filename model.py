from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wc1 = weightVariable([5,5,3, 1,16],'wc1')
    bc1 = biasVariable([16], 'bc1')

    wc2 = weightVariable([5,5,3, 16,32],'wc2')
    bc2 = biasVariable([32], 'bc2')
    
    wc3 = weightVariable([5,5,3, 32,48],'wc3')
    bc3 = biasVariable([48], 'bc3')
    
    wc4 = weightVariable([5,5,3, 48,64],'wc4')
    bc4 = biasVariable([64], 'bc4')

    wf1 = weightVariable([2304, 4096], 'wf1')
    bf1 = biasVariable([4096], 'bf1')

    wf2 = weightVariable([4096, 8192], 'wf2')
    bf2 = biasVariable([8192], 'bf2')

    wf3 = weightVariable([8192, 10], 'wf3')
    bf3 = biasVariable([10], 'bf3')

# model scratchpad
# [-1, 48, 64, 3, 1] - image
# [-1, 24, 32, 3, 16] - h1
# [-1, 12, 16, 3, 32] - h2
# [-1, 6, 8, 3, 48] - h3
# [-1, 3, 4, 3, 64] - h4

# [-1, 2304] - h4_flat
# [-1, 4096] - f1
# [-1, 8192] - f2
# [-1, 10] - f3 - flat graph to be returned

# this function returns the placeholders for inputs and targets
def getPlaceHolders():
    # the imgSize list is flipped because height and width of image are flipped when
    # converted into a numpy array
    image = tf.placeholder(tf.float32, shape=[None, imgSize[1], imgSize[0], 3, 1])
    graph_target = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    return [image, graph_target, keep_prob]

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(image, keep_prob):
    h1 = tf.nn.relu(conv3d(image, wc1) + bc1)
    h2 = tf.nn.relu(conv3d(h1, wc2) + bc2)
    h3 = tf.nn.relu(conv3d(h2, wc3) + bc3)
    h4 = tf.nn.relu(conv3d(h3, wc4) + bc4)

    h4_flat = tf.reshape(h4, [-1, 2304])
    
    f1 = tf.nn.relu(tf.matmul(h4_flat, wf1) + bf1)

    f1_drop = tf.nn.dropout(f1, keep_prob)

    f2 = tf.nn.relu(tf.matmul(f1_drop, wf2) + bf2)
    f3 = tf.nn.sigmoid(tf.matmul(f2, wf3) + bf3)

    return f3

def getGraph(vector):
    offset = tf.abs(vector - 0.1)
    # return tf.floor(2*offset)
    return tf.floor(2 * vector)

# this method returns the loss tensor
def loss(vector, graph_true):
    # return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(vector, graph_true))
    return tf.reduce_sum(tf.square(graph_true - vector))
    # graph = getGraph(vector)
    # target_sum = tf.reduce_sum(graph_true)
    # graph_sum = tf.reduce_sum(vector)
    # absDiff = 4*tf.abs(vector - graph_true)

    # t = tf.nn.sigmoid(graph_sum - target_sum)

    # maskZeros = graph_true
    # maskOnes = 1 - graph_true

    # error_ones = tf.reduce_mean(tf.mul(absDiff, maskZeros))
    # error_zeros = tf.reduce_mean(tf.mul(absDiff, maskOnes))

    # error = (t*error_zeros) + ((1-t)*error_ones) + tf.abs(graph_sum - target_sum)/20

    # return error

# this function returns the accuracy tensor
def accuracy(graph, graph_true):
    correctness = tf.equal(graph, graph_true)
    acc = 100*tf.reduce_mean(tf.cast(correctness, tf.float32))
    return acc

# this function returns the training step tensor
def getOptimStep(vector, target):
    lossTensor = loss(vector, target)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(lossTensor)
    return optim