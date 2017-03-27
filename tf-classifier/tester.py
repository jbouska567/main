#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import multilayer_perceptron as mp


# input image parameters
image_size_x = 960 #480
image_size_y = 540 #270
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
n_hidden_1 = 24 # 1st layer number of features #24
n_hidden_2 = 16 # 2nd layer number of features  #16
n_input = image_size_x * image_size_y * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)


data_path = "~/projects/tensorflow/pokus1/data"

x = tf.placeholder(tf.float32, shape=(None, n_input))

# Construct model
multilayer_perceptron = mp.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_classes)
pred = multilayer_perceptron.get_model(x)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess, "./model")


    p = subprocess.Popen(["find %s -type f" % (data_path)], stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    files = output.split()
    for filename in files:
      image = Image.open(filename)
      npi = np.array(image)
      npi = npi.reshape(n_input)
      
      cl = sess.run(tf.argmax(pred, 1), feed_dict={x: [npi]})
      print cl
      print filename
