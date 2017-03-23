from model import *
from ops import *

image, target, keep_prob = getPlaceHolders()
vector = interpret(image, keep_prob)
optim = getOptimStep(vector, target)
graph = getGraph(vector)
# graph = tf.round(tf.sigmoid(tf.abs(vector)))
accuracy = accuracy(graph, target)
lossVal = loss(vector, target)

data = dataset('data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loadModel(sess, model_save_path[0])
    # loadModel(sess, model_save_path[1])

    testBatch = data.test_batch(21)
    testBatch = data.test_batch(5)
    acc, lval, graph_out, vec = sess.run([accuracy, lossVal, graph, vector], feed_dict={
        image: testBatch[0],
        target: testBatch[1],
        keep_prob:1.0
    })

    # print(vec)

    g_sum = int(np.sum(graph_out))
    t_sum = int(np.sum(testBatch[1]))

    i = 0
    print(vec[i])
    print(testBatch[1][i])
    print(graph_out[i].astype(np.uint8))
    print('Acc: %.2f; L: %.2f; Sums: %s/%s%s'%(acc, lval,g_sum,t_sum,' '*40))