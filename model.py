from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wf1 = weightVariable([9216, 8192], 'wf1')
    bf1 = biasVariable([8192], 'bf1')
    wf2 = weightVariable([8192, 6400], 'wf2')
    bf2 = biasVariable([6400], 'bf2')
    wf3 = weightVariable([6400, 4096], 'wf3')
    bf3 = biasVariable([4096],'bf3')
    wf4 = weightVariable([4096, 2048], 'wf4')
    bf4 = biasVariable([2048], 'bf4')
    wf5 = weightVariable([2048, 1024], 'wf5')
    bf5 = biasVariable([1024], 'bf5')
    wf6 = weightVariable([1024, 512],'wf6')
    bf6 = biasVariable([512],'bf6')
    wf7 = weightVariable([512, 10], 'wf7')
    bf7 = biasVariable([10], 'bf7')

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
    im_flat = tf.reshape(image, [-1, 9216])

    h1 = tf.nn.sigmoid(tf.matmul(im_flat, wf1) + bf1)
    h2 = tf.nn.sigmoid(tf.matmul(h1, wf2) + bf2)
    h3 = tf.nn.sigmoid(tf.matmul(h2, wf3) + bf3)
    h4 = tf.nn.sigmoid(tf.matmul(h3, wf4) + bf4)
    h5 = tf.nn.sigmoid(tf.matmul(h4, wf5) + bf5)

    h5_drop = tf.nn.dropout(h5, keep_prob)

    h6 = tf.nn.sigmoid(tf.matmul(h5_drop, wf6) + bf6)
    h7 = tf.nn.sigmoid(tf.matmul(h6, wf7) + bf7)

    return h7

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