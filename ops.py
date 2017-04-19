import tensorflow as tf
import numpy as np
import os
import pickle
import time
from PIL import Image

# some global params
# the directory to which teh results will be saved
# imgSize = [96,128]
imgSize = [32,32]
spaceSize = [32,32]
batch_size = 50
# folder to save the results in
resDir = 'results/'
# folder to log the training progress in
log_dir  = 'train_log/6/'

learning_rate = 1e-4
# below is the coefficient for l2 loss
alpha = 0.002

model_save_path = ['savedModels/model_1.ckpt',
                    'savedModels/model_2.ckpt']

# in every traning example, these will be the names of the 5 spaces
nameList = ['red',
            'green',
            'yellow',
            'white']
# calculating the number of possible connections based on the number of spaces.
con_num = int(len(nameList)*(len(nameList)-1)*0.5)
# And this dictionary provides the color as RGB tuple for a space with a certian name
colors = {
    'red':(255,0,0),
    'green':(0,255,0),
    'blue':(0,0,255),
    'yellow':(255,255,0),
    'white':(255,255,255)
}

# this method saves the model
def saveModel(sess, savePath):
    print('\n...saving the models, please wait...')
    saver = tf.train.Saver()
    saver.save(sess, savePath)
    print('Saved the model to %s'%savePath)

# this method loads the saved model
def loadModel(sess, savedPath):
    print('\n...loading the models, please wait...')
    saver = tf.train.Saver()
    saver.restore(sess, savedPath)
    print('Loaded the model from %s'%savedPath)

# weight variable
def weightVariable(shape, name):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    weight = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return weight

# bias variable
def biasVariable(shape, name):
    initializer = tf.constant_initializer(0.01)
    bias = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return bias

# 2d convolutional layer
def conv2d(x, W, strides = [1,2,2,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

# 3d convolutional layer
def conv3d(x, W, strides = [1,2,2,1,1]):
    return tf.nn.conv3d(x,W,strides = strides, padding='SAME')

# deconv layer
def deConv3d(y, w, outShape, strides=[1,2,2,2,1]):
    return tf.nn.conv3d_transpose(y, w, output_shape = outShape, strides=strides, padding='SAME')

# this is max-pooling for 3d convolutional layers
def max_pool2x2x1(x):
    return tf.nn.max_pool3d(x,ksize=[1,2,2,1,1],strides=[1,2,2,1,1],padding='SAME')

# this is a 2x2 max-pooling layer for 2d convolutional layers
def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# converts data to image
def toImage(data):
    data = np.reshape(data, [48,64])
    newData = 255*data
    # converting new data into integer format to make it possible to export it as a bitmap
    # in this case converting it into 8 bit integer
    newData = newData.astype(np.uint8)
    return Image.fromarray(newData)

# this method converts a list of images into a feed ready batch
def prepareImages(imageList, normalize = True):
    batch = []
    for img in imageList:
        arr = prepareImage(img, normalize=normalize)
        batch.append(arr)
    return batch

# this one prepares a single image and returns it as an array
def prepareImage(img, normalize = True):
    arr = np.array(img)
    arr = arr.astype(np.float32)
    if normalize:
        arr /= 255

    return arr

# this method returns the time estimate as a string
def estimate_time(startTime, totalCycles, finishedCycles):
    timePast = time.time() - startTime
    if timePast < 10:
        return '...calculating...'
    cps = finishedCycles/timePast
    if cps == 0:
        return '...calculating...'
        
    secsLeft = (totalCycles - finishedCycles)/cps

    # double slash is an integer division, returns the quotient without the decimal
    hrs, secs = secsLeft//3600, secsLeft % 3600
    mins, secs = secs//60, secs % 60

    timeStr = '%.0fh%.0fm%.0fs remaining'%(hrs, mins, secs) + ' '*10
    return timeStr

# this is the dataset object which is responsible with supplying data for training as well as
# testing purposes
class dataset:
    # the dirPath is the path of the directory
    # testFile is the name of the file with the test examples
    def __init__(self, dirPath = './', testFile='test'):
        self.dirPath = dirPath
        self.testFileName = testFile + '.pkl'
        # getting the list of files in the dirPath
        self.trainFileList = os.listdir(self.dirPath)
        # removing the test data file from the list of training data files
        if len(self.trainFileList) > 1:
            self.trainFileList.remove(self.testFileName)

        #removing any other file than pickled data files from the list
        i = 0
        while i < len(self.trainFileList):
            if not self.trainFileList[i].endswith('.pkl'):
                # print('removing %s'%self.trainFileList[i])
                del self.trainFileList[i]
                i -= 1
            
            i += 1
        
        # print(self.trainFileList)
        
        self.curFile = None
        if len(self.trainFileList) > 0:
            self.curFile = self.trainFileList[0]
        self.data = None
        self.test_data = None
        self.load_data()

        # this is the number of data samples currently loaded
        self.data_num = self.data[0].shape[0]
        self.test_data_num = self.test_data[0].shape[0]

        # this is the counter for where we are reading the data in the currently loaded set
        self.c = 0
        self.tc = 0
    
    # returns the name of the next file to be used once the current file is exhausted
    def next_file(self):
        fileNum = self.trainFileList.index(self.curFile)
        fileNum = (fileNum + 1)%len(self.trainFileList)
        nextFile = self.trainFileList[fileNum]
        return nextFile
    
    # this loads the data from the current file into the self.data
    def load_data(self, silent = False):
        with open(self.dirPath + self.curFile,'rb') as inp:
            if not silent: print('\nLoading data from %s...'%self.curFile)
            dSet = pickle.load(inp)
        
        # self.data = [np.expand_dims(np.array(dSet[0]),3), np.array(dSet[1])]
        self.data = [np.array(dSet[0]), np.array(dSet[1])]
        # self.data = dSet
        
        if self.test_data is None:
            with open(self.dirPath + self.testFileName,'rb') as inp:
                if not silent: print('\nLoading test data from %s...'%self.testFileName)
                dSet = pickle.load(inp)
            
            # self.test_data = [np.expand_dims(np.array(dSet[0]),3), np.array(dSet[1])]
            self.test_data = [np.array(dSet[0]), np.array(dSet[1])]
            # self.test_data = dSet
        if not silent: print('\nDataset in %s is successfully loaded\n'%self.dirPath)
    
    # this returns the next batch of the size - size
    def next_batch(self, size):
        if self.c + size >= self.data_num:
            end = self.data_num
            dataNeeded = size-(end-self.c)

            batch1 = [self.data[0][self.c: end], self.data[1][self.c: end]]

            # now getting the remaining data from the next file
            end = (self.c + size)%self.data_num
            self.c = 0
            self.curFile = self.next_file()
            self.load_data(silent = True)

            batch2 = [self.data[0][self.c: end], self.data[1][self.c: end]]
            # now joining batch 2 to the original batch
            batch = [np.concatenate((batch1[0], batch2[0]), axis=0), np.concatenate((batch1[1], batch2[1]), axis=0)]

            self.c = end

        elif self.c + size < self.data_num:
            end = self.c + size
            
            batch = [self.data[0][self.c: end], self.data[1][self.c: end]]
            self.c = end

        return batch
    
    # this fetches a test batch
    def test_batch(self, size):
        if self.tc + size >= self.test_data_num:
            end = self.test_data_num
            dataNeeded = size-(end-self.tc)

            batch1 = [self.test_data[0][self.tc: end], self.test_data[1][self.tc: end]]

            end = (self.tc + size)%self.test_data_num
            self.tc = 0

            batch2 = [self.test_data[0][self.tc: end], self.test_data[1][self.tc: end]]
            # now joining batch 2 to the original batch
            batch = [np.concatenate((batch1[0], batch2[0]), axis=0), np.concatenate((batch1[1], batch2[1]), axis=0)]

            self.tc = end

        elif self.tc + size < self.test_data_num:
            end = self.tc + size
            
            batch = [self.test_data[0][self.tc: end], self.test_data[1][self.tc: end]]
            self.tc = end
        
        return batch

# this just saves any given data to any given path
def writeToFile(data, path):
    with open(path, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

# this creates summaries for variables to be used by tensorboard
def summarize(varT):
    varName = varT.name[:-2]
    with tf.name_scope(varName):
        var_mean = tf.reduce_mean(varT)
        var_sum = tf.reduce_sum(varT)
        tf.summary.scalar('mean', var_mean)
        tf.summary.scalar('sum', var_sum)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(varT - var_mean)))
        
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(varT))
        tf.summary.scalar('min', tf.reduce_min(varT))
        tf.summary.histogram('histogram', varT)

# this returns the writer objects for training and testing
def getSummaryWriters(sess):
    train_writer = tf.summary.FileWriter(log_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + 'test')

    return [train_writer, test_writer]
# from here down is the sandbox place to check and verify the code above before using it in
# the other files