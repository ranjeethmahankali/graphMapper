import tensorflow as tf

tSize = tf.placeholder(tf.int32)
tens = tf.placeholder(tf.float32, shape=tf.pack([1,tSize]))