#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import multilayer_perceptron as mp
from preprocess_image import read_preprocess_image

#TODO dodelat do testeru clusterovani (udelat moduly)

# input image parameters
FUZZ = 10
image_div = 2
cluster_size = 10
image_size_x = 1920 / image_div
image_size_y = 1080 / image_div
channels = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features #24
n_hidden_2 = 128 # 2nd layer number of features  #16
n_input = (image_size_x / cluster_size) * (image_size_y / cluster_size) * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

data_path = "/home/pepa/projects/camera_filter/learning/diff%s-%s" % (FUZZ, image_size_x)
model_name = "model-%s-%s-%s-%s-%s" % (FUZZ, image_div, cluster_size, n_hidden_1, n_hidden_2)
model_name = model_name + "-140"

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
    print "opening model %s" % model_name
    saver.restore(sess, "./"+model_name)

    p = subprocess.Popen(["find %s -type f" % (data_path)], stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    files = output.split()
    positive_count = 0
    negative_count = 0
    positive_mismatch = 0
    negative_mismatch = 0
    print "Prediction mismatches:"
    for filename in files:
        npi, label = read_preprocess_image(filename, cluster_size)
        npi = npi.reshape(n_input)

        #print "klasifikace: %s, label: %s, image: " % (cl, label, filename)
        cl = sess.run(tf.argmax(pred, 1), feed_dict={x: [npi]})

        if label:
            positive_count = positive_count + 1
        else:
            negative_count = negative_count + 1
        if cl != label:
            print "Prediction: %s, label: %s, test file: %s" % (cl, label, filename)
            if label:
                positive_mismatch = positive_mismatch + 1
            else:
                negative_mismatch = negative_mismatch + 1

    print "%s/%s true and %s/%s false mismatches" % (positive_mismatch, positive_count, negative_mismatch, negative_count)
    print "Accuracy on positive alarm: %0.4f" % (float(positive_count - positive_mismatch) / positive_count)
    print "Accuracy on negative alarm: %0.4f" % (float(negative_count - negative_mismatch) / negative_count)
