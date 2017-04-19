from inceptionModel import *
from ops import *

bottleneck, target, keep_prob = getPlaceHolders()
vector, graph = interpret(bottleneck, keep_prob)
optim, lossVal = getOptimStep(vector, graph, target)
accuracy = accuracy(graph, target)

data = dataset('inception_data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loadModel(sess, model_save_path[1])
    # loadModel(sess, model_save_path[1])

    # testBatch = data.test_batch(21)
    for i in range(10000//50):
        testBatch = data.test_batch(50)
        acc, lval, graph_out, vec = sess.run([accuracy, lossVal, graph, vector], feed_dict={
            bottleneck: testBatch[0],
            target: testBatch[1],
            keep_prob:1.0
        })
        
        g_sum = int(np.sum(graph_out))
        t_sum = int(np.sum(testBatch[1]))

        print('Acc: %.2f; L: %.2f; Sums: %s/%s%s'%(acc, lval,g_sum,t_sum,' '*40))