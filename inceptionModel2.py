from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wf1 = weightVariable([2048, 10], 'wf1')
    bf1 = biasVariable([10], 'bf1')
    # wf4 = weightVariable([1024, 10], 'wf4')
    # bf4 = biasVariable([10], 'bf4')

# adding summaries to all the above variables
all_vars = tf.trainable_variables()
varList = [v for v in all_vars if 'vars' in v.name]

# this function returns the placeholders for inputs and targets
def getPlaceHolders():
    # the imgSize list is flipped because height and width of image are flipped when
    # converted into a numpy array
    bottleneck = tf.placeholder(tf.float32, shape=[None, 2048])
    graph_target = tf.placeholder(tf.float32, shape=[None, 10])

    return [bottleneck, graph_target]

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(bottleneck):
    y = tf.nn.sigmoid(tf.matmul(bottleneck, wf1) + bf1)
    # adding summaries for the final output
    summarize(y)
    return [y, getGraph(y, True)]

# this one converts the output of the network into 1 or 0 format
def getGraph(vector, toSummarize = False):
    # defined this as a separate function just in case I change the network in the future
    # and will need to do more than just rounding to map its output to 0 or 1
    # small constant for numerical stability
    graph = tf.round(vector, name='graph_out')
    
    if toSummarize:
        summarize(graph)

    return graph

# this method returns the loss tensor
def loss(vector, graph_true):
    # return tf.reduce_sum(tf.abs(graph_true - vector))
    # return tf.reduce_sum(tf.square(graph_true - vector))
    epsilon = 1e-9
    cross_entropy = -((graph_true * tf.log(vector + epsilon)) + ((1-graph_true)*tf.log(1-vector+epsilon)))
    
    # adding this to summaries
    ce_by_example = tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy')
    summarize(ce_by_example)

    ce_loss = tf.reduce_sum(ce_by_example)

    return ce_loss

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
    # lossTensor = loss_custom(vector, graph, target)
    lossTensor = loss(vector,target)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(lossTensor)
    return [optim, lossTensor]