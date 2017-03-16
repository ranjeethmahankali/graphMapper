from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wc1 = weightVariable([5,5,3,16], 'wc1')
    bc1 = biasVariable([16],'bc1')
    wc2 = weightVariable([5,5,16,32], 'wc2')
    bc2 = biasVariable([32],'bc2')
    wc3 = weightVariable([5,5,32,64],'wc3')
    bc3=  biasVariable([64],'bc3')
    wc4 = weightVariable([5,5,64,256],'wc4')
    bc4 =  biasVariable([256],'bc4')

    wf1 = weightVariable([3072, 8192], 'wf1')
    bf1 = biasVariable([8192], 'bf1')
    wf2 = weightVariable([8192, 4096], 'wf2')
    bf2 = biasVariable([4096], 'bf2')
    wf3 = weightVariable([4096, 2048], 'wf3')
    bf3 = biasVariable([2048], 'bf3')
    wf4 = weightVariable([2048, 10], 'wf4')
    bf4 = biasVariable([10], 'bf4')


# model scratch pad
# starting with a three channel image of 3 channels
# [None, 48,64,3] - image
# [None, 24, 32, 16] - h0
# [None, 12, 16, 32] - h1
# [None, 6, 8, 64] - h2
# [None, 3, 4, 256] - h3

# we then flatten h3
# [None, 3072] - h3_flat
# [None, 8192] - f0
# [None, 4096] - f1
# [None, 2048] - f2
# [None, 10] - f3 - this is whatever the size of the flattned graph would be for 5 spaces, it is 10

# this function returns the placeholders for inputs and targets
def getPlaceHolders():
    # the imgSize list is flipped because height and width of image are flipped when
    # converted into a numpy array
    image = tf.placeholder(tf.float32, shape=[None, imgSize[1], imgSize[0], 3])
    graph_target = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    return [image, graph_target, keep_prob]

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(image, keep_prob):
    h0 = tf.nn.sigmoid(conv2d(image, wc1) + bc1)
    h1 = tf.nn.sigmoid(conv2d(h0, wc2) + bc2)
    h2 = tf.nn.sigmoid(conv2d(h1, wc3) + bc3)
    h3 = tf.nn.sigmoid(conv2d(h2, wc4) + bc4)
    
    h3_flat = tf.reshape(h3, [-1, 3072])
    
    f0 = tf.nn.sigmoid(tf.matmul(h3_flat, wf1) + bf1)
    f1 = tf.nn.sigmoid(tf.matmul(f0, wf2) + bf2)
    f1_drop = tf.nn.dropout(f1, keep_prob)
    f2 = tf.nn.sigmoid(tf.matmul(f1_drop, wf3) + bf3)
    f3 = tf.nn.sigmoid(tf.matmul(f2, wf4) + bf4)

    # f3 is th predicted vector which has floating point numbers
    return f3

def getGraph(vector):
    offset = tf.abs(vector - 0.1)
    # return tf.floor(2*offset)
    return tf.floor(2 * vector)

# this method returns the loss tensor
def loss(vector, graph_true):
    # return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(vector, graph_true))
    # return tf.reduce_mean(tf.square(graph_true - vector))
    graph = getGraph(vector)
    target_sum = tf.reduce_sum(graph_true)
    graph_sum = tf.reduce_sum(vector)
    absDiff = 4*tf.abs(vector - graph_true)

    t = tf.nn.sigmoid(graph_sum - target_sum)

    maskZeros = graph_true
    maskOnes = 1 - graph_true

    error_ones = tf.reduce_mean(tf.mul(absDiff, maskZeros))
    error_zeros = tf.reduce_mean(tf.mul(absDiff, maskOnes))

    error = (t*error_zeros) + ((1-t)*error_ones) + tf.abs(graph_sum - target_sum)/20

    return error

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