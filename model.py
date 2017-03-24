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

    wf1 = weightVariable([2304, 2304], 'wf1')
    bf1 = biasVariable([2304], 'bf1')

    wf2 = weightVariable([2304, 10], 'wf2')
    bf2 = biasVariable([10], 'bf2')

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

# this is a combination of convolutional layer and pooling layer
def conv_pool(x, W, b):
    conv = tf.nn.relu(conv3d(x, W, strides=[1,1,1,1,1]) + b)
    pool = max_pool2x2x1(conv)

    return pool

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(image, keep_prob):
    h1 = conv_pool(image, wc1, bc1)
    h2 = conv_pool(h1, wc2, bc2)
    h3 = conv_pool(h2, wc3, bc3)
    h4 = conv_pool(h3, wc4, bc4)

    h4_flat = tf.reshape(h4, [-1, 2304])
    
    f1 = tf.nn.relu(tf.matmul(h4_flat, wf1) + bf1)

    f1_drop = tf.nn.dropout(f1, keep_prob)    
    f2 = tf.nn.sigmoid(tf.matmul(f1_drop, wf2) + bf2)

    # this is the output that is used to calculate error
    # output = (1 + f3)/2

    return f2

def getGraph(vector):
    return tf.round(vector)
    # mean = tf.reduce_mean(vector, axis=1, keep_dims=True)
    # shift = vector - mean
    # norm = tf.nn.sigmoid(shift)
    # return tf.round(norm)

# this method returns the loss tensor
def loss(vector, graph_true):
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(vector, graph_true))
    # return tf.reduce_sum(tf.abs(graph_true - vector))
    # return tf.reduce_sum(tf.square(graph_true - vector))
    scale = 0.1
    graph = getGraph(vector)
    target_sum = tf.reduce_sum(graph_true)
    graph_sum = tf.reduce_sum(vector)
    absDiff = tf.abs(vector - graph_true)/scale

    # return tf.reduce_sum(absDiff)

    t = tf.nn.sigmoid(graph_sum - target_sum)

    maskZeros = graph_true
    maskOnes = 1 - graph_true

    error_ones = tf.reduce_mean(tf.mul(absDiff, maskZeros))
    error_zeros = tf.reduce_mean(tf.mul(absDiff, maskOnes))
    
    error = (t*error_zeros) + ((1-t)*error_ones)

    # now implementing l2 loss
    all_vars = tf.trainable_variables()
    varList = [v for v in all_vars if 'vars' in v.name]
    l2_loss = 0
    for v in varList:
        l2_loss += tf.nn.l2_loss(v)*alpha

    return (error + l2_loss)

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