from model import *
import sys

image, target = getPlaceHolders()
vector = interpret(image)
optim = getOptimStep(vector, target)
graph = getGraph(vector)
accuracy = accuracy(graph, target)
lossVal = loss(vector, target)

data = dataset('data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # loadModel(sess, model_save_path[0])
    # loadModel(sess, model_save_path[1])

    cycles = 4000
    testStep = 50
    saveStep = 2000
    startTime = time.time()
    try:
        for i in range(cycles):
            batch = data.next_batch(batch_size)
            _ = sess.run(optim, feed_dict={
                image: batch[0],
                target: batch[1]
            })

            timer = estimate_time(startTime, cycles, i)
            pL = 10 # this is the length of the progress bar to be displayed
            pNum = i % pL
            pBar = '#'*pNum + ' '*(pL - pNum)

            sys.stdout.write('...Training...|%s|-(%s/%s)- %s\r'%(pBar, i, cycles, timer))

            if i % testStep == 0:
                testBatch = data.test_batch(batch_size)
                acc, lval = sess.run([accuracy, lossVal], feed_dict={
                    image: testBatch[0],
                    target: testBatch[1]
                })
                
                # tracker helps to compare the data being printed to previous run with same 
                # training examples
                tracker = (i/testStep)%(1000/batch_size)
                print('%02d Accuracy: %.2f; Loss: %.2f%s'%(tracker, acc, lval,' '*50))
        
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