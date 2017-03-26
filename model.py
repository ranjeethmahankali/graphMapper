from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wc1 = weightVariable([5,5, 1,32],'wc1')
    bc1 = biasVariable([32], 'bc1')

    wc2 = weightVariable([5,5, 32,64],'wc2')
    bc2 = biasVariable([64], 'bc2')
    
    wc3 = weightVariable([5,5, 64,128],'wc3')
    bc3 = biasVariable([128], 'bc3')
    
    wc4 = weightVariable([5,5, 128,256],'wc4')
    bc4 = biasVariable([256], 'bc4')

    wf1 = weightVariable([3072, 4096], 'wf1')
    bf1 = biasVariable([4096], 'bf1')

    wf2 = weightVariable([4096, 2048], 'wf2')
    bf2 = biasVariable([2048], 'bf2')

    wf3 = weightVariable([2048, 2048], 'wf3')
    bf3 = biasVariable([2048], 'bf3')

    wf4 = weightVariable([2048, 10], 'wf4')
    bf4 = biasVariable([10], 'bf4')

# adding summaries to all the above variables
# all_vars = tf.trainable_variables()
# varList = [v for v in all_vars if 'vars' in v.name]
# for v in varList:
#     summarize(v)

# model scratchpad
# [-1, 48, 64, 1] - image
# [-1, 24, 32, 32] - h1
# [-1, 12, 16, 64] - h2
# [-1, 6, 8, 128] - h3
# [-1, 3, 4, 256] - h4

# [-1, 3072] - h4_flat
# [-1, 4096] - f1
# [-1, 2048] - f2
# [-1, 2048] - f3
# [-1, 10] - f4 - flat graph to be returned

# this function returns the placeholders for inputs and targets
def getPlaceHolders():
    # the imgSize list is flipped because height and width of image are flipped when
    # converted into a numpy array
    image = tf.placeholder(tf.float32, shape=[None, imgSize[1], imgSize[0], 1])
    graph_target = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    return [image, graph_target, keep_prob]

# this is a combination of convolutional layer and pooling layer
def conv_pool(x, W, b):
    conv = tf.nn.relu(conv2d(x, W, strides=[1,1,1,1]) + b)
    pool = max_pool2x2(conv)

    return pool

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(image, keep_prob):
    h1 = conv_pool(image, wc1, bc1)
    h2 = conv_pool(h1, wc2, bc2)
    h3 = conv_pool(h2, wc3, bc3)
    h4 = conv_pool(h3, wc4, bc4)

    h4_flat = tf.reshape(h4, [-1, 3072])
    h4_flat_drop = tf.nn.dropout(h4_flat, keep_prob)

    f1 = tf.nn.relu(tf.matmul(h4_flat_drop, wf1) + bf1)
    f2 = tf.nn.relu(tf.matmul(f1, wf2) + bf2)
    f3 = tf.nn.relu(tf.matmul(f2, wf3) + bf3)
    f4 = tf.nn.sigmoid(tf.matmul(f3, wf4) + bf4, name = 'output')

    # adding summaries for the final output
    summarize(f4)

    return [f4, getGraph(f4)]

# this one converts the output of the network into 1 or 0 format
def getGraph(vector, toSummarize = False):
    # defined this as a separate function just in case I change the network in the future
    # and will need to do more than just rounding to map its output to 0 or 1
    # small constant for numerical stability
    epsilon = 1e-9
    graph = tf.floor(2*(vector-epsilon))
    if toSummarize: 
        summarize(graph)

    return graph

# this method returns the loss tensor
def loss(vector, graph_true):
    # return tf.reduce_sum(tf.abs(graph_true - vector))
    # return tf.reduce_sum(tf.square(graph_true - vector))
    epsilon = 1e-9
    cross_entropy = (graph_true * tf.log(vector + epsilon)) + ((1-graph_true)*tf.log(1-vector+epsilon))
    
    # adding this to summaries
    ce_by_example = -tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy')
    summarize(ce_by_example)

    ce_loss = -tf.reduce_sum(cross_entropy)

    return ce_loss

# this is the custom loss that I used for the voxel models
def loss_custom(m, g, gTrue):
    # this is the absolute difference between the two tensors
    absDiff = tf.abs(m - gTrue)
    
    scale = 2
    g_sum = tf.reduce_sum(g) / scale
    gTrue_sum = tf.reduce_sum(gTrue) / scale

    maskZeros = gTrue
    maskOnes = 1 - gTrue

    # this is the error for not filling the voxels that are supposed to be filled
    error_ones = tf.reduce_sum(tf.mul(absDiff, maskZeros))
    # this is the error for filling the voxels that are not supposed to be filled
    error_zeros = tf.reduce_sum(tf.mul(absDiff, maskOnes))

    # this is the dynamic factor representing how much you care about which error
    factor = tf.nn.sigmoid(g_sum - gTrue_sum)

    error = (factor*error_zeros) + ((1-factor)*error_ones)

    # sending this to summary
    with tf.name_scope('loss_params'):
        tf.summary.scalar('toggleFactor', t)
        tf.summary.scalar('e_zeros', (t*error_zeros))
        tf.summary.scalar('e_ones', ((1-t)*error_ones))
        tf.summary.scalar('e_combined', error)

    # now implementing l2 loss
    # l2_loss = 0
    # for v in varList:
    #     l2_loss += tf.nn.l2_loss(v)*alpha

    # return (error + l2_loss)
    return error

# this function returns the accuracy tensor
def accuracy(graph, graph_true):
    correctness = tf.equal(graph, graph_true)
    acc_norm = tf.cast(correctness, tf.float32)
    acc = tf.multiply(acc_norm, 100)

    acc_by_example = tf.reduce_mean(acc, axis=1,name='accuracy')

    summarize(acc_by_example)

    return tf.reduce_mean(acc_by_example)

# this function returns the training step tensor
def getOptimStep(vector, graph, target):
    lossTensor = loss_custom(vector, graph, target)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(lossTensor)
    return [optim, lossTensor]