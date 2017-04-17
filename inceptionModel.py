from ops import *
# from dataGen import space
import tensorflow as tf

with tf.variable_scope('vars'):
    wf1 = weightVariable([2048, 3072], 'wf1')
    bf1 = biasVariable([3072], 'bf1')

    wf2 = weightVariable([3072, 2048], 'wf2')
    bf2 = biasVariable([2048], 'bf2')

    wf3 = weightVariable([2048, con_num], 'wf3')
    bf3 = biasVariable([con_num], 'bf3')

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
    graph_target = tf.placeholder(tf.float32, shape=[None, con_num])
    keep_prob = tf.placeholder(tf.float32)

    return [bottleneck, graph_target, keep_prob]

# this interprets the image and returns the tensor corresponding to a flattened graph
def interpret(bottleneck, keep_prob):
    drop_input = tf.nn.dropout(bottleneck, keep_prob)
    f1 = tf.nn.relu(tf.matmul(drop_input, wf1) + bf1)
    f1_drop = tf.nn.dropout(f1, keep_prob)
    f2 = tf.nn.tanh(tf.matmul(f1_drop, wf2) + bf2)
    f3 = tf.nn.sigmoid(tf.matmul(f2, wf3) + bf3, name='output')
    # f4 = tf.nn.sigmoid(tf.matmul(f3, wf4) + bf4, name='output')

    # adding summaries for the final output
    summarize(f3)

    return [f3, getGraph(f3, True)]

# this one converts the output of the network into 1 or 0 format
def getGraph(vector, toSummarize = False):
    # defined this as a separate function just in case I change the network in the future
    # and will need to do more than just rounding to map its output to 0 or 1
    # small constant for numerical stability
    epsilon = 1e-9
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

    # now implementing regularizations
    l2_loss = 0
    for v in varList:
        l2_loss += tf.nn.l2_loss(v)

    l2_loss *= alpha
    total_loss = lossTensor + l2_loss

    # adding to the summary
    with tf.name_scope('loss_params'):
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('total_loss', total_loss)

    optim = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return [optim, total_loss]