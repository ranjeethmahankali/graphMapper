import inceptionPrep as ip
from ops import *
from spaceGraph_ext import *

bottleneck_vals = []

fileNum = len(os.listdir(resDir))
for i in range(fileNum):
    fileName = resDir+'%s.png'%i
    img_arr = prepareImage(Image.open(fileName), normalize=False)

    bottleneck_val = ip.run_bottleneck_on_image(ip.sess_inception, 
                                                    img_arr, 
                                                    ip.jpeg_data_tensor, 
                                                    ip.bottleneck_tensor)

    # bottleneck_val = np.expand_dims(bottleneck_val, axis=0)
    bottleneck_vals.append(bottleneck_val)

ip.sess_inception.close()

from inceptionModel import *

bottleneck, target, keep_prob = getPlaceHolders()
vector, graph = interpret(bottleneck, keep_prob)
optim, lossVal = getOptimStep(vector, graph, target)
accuracy = accuracy(graph, target)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loadModel(sess, model_save_path[1])
    flat_graph_list = sess.run(graph, feed_dict={
        bottleneck: bottleneck_vals,
        keep_prob:1.0
    })

    sampleSpace = space('sample', nameList, default_coords)
    for n in range(len(flat_graph_list)):
        flatGraph = flat_graph_list[n]
        for i in range(len(flatGraph)):
            if flatGraph[i]==0:continue
            a,b = sampleSpace.getConSpaces(i)
            print(nameList[a], nameList[b])

        print('==================================')