import tensorflow as tf
import numpy as np
import os.path
import hashlib
import sys
import tarfile
import time

from io import BytesIO
from six.moves import urllib
from tensorflow.python.platform import gfile
from PIL import Image

# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
model_dir = 'inceptionGraph'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 64
MODEL_INPUT_HEIGHT = 48
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

def maybe_download_and_extract():
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                            (filename,
                            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.
    Returns:
    Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# model_fn = 'inceptionGraph/tensorflow_inception_graph.pb'
model_fn = os.path.join(model_dir, 'classify_image_graph_def.pb')
maybe_download_and_extract()


graph_inception = tf.Graph()
sess_inception = tf.InteractiveSession(graph = graph_inception)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[
                BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                RESIZED_INPUT_TENSOR_NAME]))

print('inception imported succesfully!')

# with tf.Session() as sess:
#     img = Image.open('test.jpg')
#     # buffer = BytesIO()
#     # img.save(buffer, format='JPEG')
#     # image_data = buffer.getvalue()
#     arr = np.array(img)
#     image_data = arr.astype(np.float32)
#     # image_data = gfile.FastGFile('test.jpg', 'rb').read()
#     startTime = time.time()
#     bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
#     delta = time.time() - startTime
#     bottleneck_values = np.squeeze(bottleneck_values)
#     print(bottleneck_values)
#     print(delta)

#     startTime = time.time()
#     bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
#     delta = time.time() - startTime
#     bottleneck_values = np.squeeze(bottleneck_values)
#     print(bottleneck_values)
#     print(delta)

#     startTime = time.time()
#     bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
#     delta = time.time() - startTime
#     bottleneck_values = np.squeeze(bottleneck_values)
#     print(bottleneck_values)
#     print(delta)