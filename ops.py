import tensorflow as tf
import numpy as np
import os
import pickle
import time
from PIL import Image

# some global params
# the directory to which teh results will be saved
# imgSize = [96,128]
imgSize = [64,48]
spaceSize = [64,48]
batch_size = 5
# resDir = 'results/'
resDir = 'results/'
learning_rate = 1e-5
model_save_path = ['savedModels/model_1.ckpt',
                    'savedModels/model_2.ckpt']

# in every traning example, these will be the names of the 5 spaces
nameList = ['red',
            'green',
            'blue',
            'yellow',
            'white']
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
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    weight = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return weight

# bias variable
def biasVariable(shape, name):
    initializer = tf.constant_initializer(0.1)
    bias = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return bias

# 2d convolutional layer
def conv2d(x, W, strides = [1,2,2,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

# deconv layer
def deConv3d(y, w, outShape, strides=[1,2,2,2,1]):
    return tf.nn.conv3d_transpose(y, w, output_shape = outShape, strides=strides, padding='SAME')

# converts data to image
def toImage(data):
    data = np.reshape(data, imgSize)
    newData = 255*data
    # converting new data into integer format to make it possible to export it as a bitmap
    # in this case converting it into 8 bit integer
    newData = newData.astype(np.uint8)
    return Image.fromarray(newData)

# this method converts a list of images into a feed ready batch
def prepareImages(fileList):
    batch = []
    for fileName in fileList:
        img = Image.open(fileName).convert("L")
        arr prepareImage(img)
        batch.append(arr)
    return np.array(batch)

# this one prepares a single image and returns it as an array
def prepareImage(img):
    arr = np.array(img)
    arr = np.expand_dims(arr, 2)
    arr = arr.astype(np.float32)
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
        
        # self.data = [np.expand_dims(np.array(dSet[0]),3), np.expand_dims(np.array(dSet[1]), 4)]
        self.data = dSet
        
        if self.test_data is None:
            with open(self.dirPath + self.testFileName,'rb') as inp:
                if not silent: print('\nLoading test data from %s...'%self.testFileName)
                dSet = pickle.load(inp)
            
            # self.test_data = [np.expand_dims(np.array(dSet[0]),3), np.expand_dims(np.array(dSet[1]), 4)]
            self.test_data = dSet
    
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

# this method converts the np arrays back into normal lists to be used in rhinopython
def voxToRhino(vox_np_data):
    vox = vox_np_data.squeeze(axis=4).tolist()
    return vox
# This method saves a batch of results as images and vox lists to compare
def saveResults(batch, fileName='vox.pkl', version = 2, saveImages = True):
    # batch is a list of len 2 batch[0] is the images and batch[1] is the voxels
    # now convertiing the voxels for rhino and pickling them
    vox = voxToRhino(batch[1])
    with open(resDir+fileName, 'wb') as output:
        pickle.dump(vox, output, protocol = version)
    
    if saveImages:
        imgNum = batch[0].shape[0]
        for i in range(imgNum):
            img  = toImage(batch[0][i:i+1])
            img.save(resDir+'%s.png'%i)
    
    print(' ... results saved')
# from here down is the sandbox place to check and verify the code above before using it in
# the other files