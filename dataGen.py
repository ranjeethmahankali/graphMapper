from spaceGraph_ext import *
from inceptionPrep import *
# the main logic begins here, if any

coords = {'pt':[], 'x0':0,'x1':imgSize[0],'y0':0,'y1':imgSize[1]}

# the folder to which dataset will be saved
dataDir = 'inception_data/'
# number of files
fileNum = 50
# number of training configurations per file that we want
dataNum = 200
# number of door variations in each configuration
doorVarNum = 10
# total size of the dataset is fileNum * dataNum * doorVarNum
# this is the number of files that is already in the dataset folder
filesCreated = 0

startTime = time.time()
for n in range(filesCreated, filesCreated+fileNum):
    im_data = list()
    graph_data = list()
    i = 0
    while i < dataNum:
        sample = space('sample', nameList, coords)
        sample.populatePts()
        sample.makeWalls()
        if len(sample.c) < len(nameList):
            # this is when 3 or more points lie on the walls and
            # mess everything up, we just ignore and repeat since
            # this is a rare random occurence
            continue
        sample.splitWalls()

        for j in range(doorVarNum):
            sample.removeDoors()
            sample.makeRandDoors()

            img = sample.render()
            # img.show()
            fileName = 'images/%s_%s_%s.png'%(n,i,j)
            im_arr = prepareImage(img, normalize=False)

            bottleneck_values = run_bottleneck_on_image(sess, 
                                                        im_arr, 
                                                        jpeg_data_tensor, 
                                                        bottleneck_tensor)

            flat_graph = sample.getFlatGraph()
            # print(flat_graph)
            
            im_data.append(bottleneck_values)
            graph_data.append(flat_graph)
        
        timeLeft = estimate_time(startTime, fileNum*dataNum, ((n-filesCreated)*dataNum)+i)
        sys.stdout.write('%s examples generated...%s\r'%(i+1, timeLeft))
        
        i += 1

    savePath = dataDir+str(n)+'.pkl'
    # savePath = dataDir + 'test.pkl'
    writeToFile([im_data, graph_data], savePath)
    print('\nSaved to %s'%savePath)
    print('%s'%('-'*25))

# sample.printSpace()
# print(sample.getFlatGraph())
# img.show()
# img = Image.open('test.jpg')
    # # buffer = BytesIO()
    # # img.save(buffer, format='JPEG')
    # # image_data = buffer.getvalue()
    # arr = np.array(img)
    # image_data = arr.astype(np.float32)
    # # image_data = gfile.FastGFile('test.jpg', 'rb').read()
    # startTime = time.time()
    # bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    # delta = time.time() - startTime
    # bottleneck_values = np.squeeze(bottleneck_values)