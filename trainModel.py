from model import *
import sys
from ops import *

image, target, keep_prob = getPlaceHolders()
vector = interpret(image, keep_prob)
optim = getOptimStep(vector, target)
graph = getGraph(vector)
accuracy = accuracy(graph, target)
lossVal = loss(vector, target)

data = dataset('data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loadModel(sess, model_save_path[0])
    # loadModel(sess, model_save_path[1])

    cycles = 15000
    testStep = 200
    saveStep = 1000
    startTime = time.time()
    test_batch_size = batch_size*10
    try:
        for i in range(cycles):
            batch = data.next_batch(batch_size)
            _ = sess.run(optim, feed_dict={
                image: batch[0],
                target: batch[1],
                keep_prob:0.5
            })

            timer = estimate_time(startTime, cycles, i)
            pL = 10 # this is the length of the progress bar to be displayed
            pNum = i % pL
            pBar = '#'*pNum + ' '*(pL - pNum)

            sys.stdout.write('...Training...|%s|-(%s/%s)- %s\r'%(pBar, i, cycles, timer))

            if i % testStep == 0:
                testBatch = data.test_batch(test_batch_size)
                acc, lval, graph_out, vec = sess.run([accuracy, lossVal, graph, vector], feed_dict={
                    image: testBatch[0],
                    target: testBatch[1],
                    keep_prob:1.0
                })
                
                g_sum = int(np.sum(graph_out))
                t_sum = int(np.sum(testBatch[1]))
                # tracker helps to compare the data being printed to previous run with same 
                # training examples
                tracker = (i/testStep)%(1000/test_batch_size)
                # print(testBatch[0][0])
                # print(vec[0], testBatch[1][0])
                print('%02d Acc: %.2f; L: %.2f; Sums: %s/%s%s'%(tracker, acc, lval,g_sum,t_sum,' '*40))
        
        # now saving the trained model every 1500 cycles
            if i % saveStep == 0 and i != 0:
                saveModel(sess, model_save_path[0])
        
        # saving the model in the end
        saveModel(sess, model_save_path[0])
    # if the training is interrupted from keyboard (ctrl + c)
    except KeyboardInterrupt:
        print('')
        print('You interrupted the training process')
        decision = input('Do you want to save the current model before exiting? (y/n):')

        if decision == 'y':
            saveModel(sess, model_save_path[0])
        elif decision == 'n':
            print('\n...Model not saved...')
            pass
        else:
            print('\n...Invalid input. Model not saved...')
            pass