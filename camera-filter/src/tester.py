#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from lib.multilayer_perceptron import MultilayerPerceptron
import lib.preprocess_image as pi

#TODO promenne z yml configu

# input image parameters
image_div = 2
cluster_size = 30

image_size_x = 1920 / image_div
image_size_y = 1080 / image_div
channels = 1

# Network Parameters
n_hidden_1 = 64 # 1st layer number of features #24
n_hidden_2 = 32 # 2nd layer number of features  #16
n_input = (image_size_x / cluster_size) * (image_size_y / cluster_size) * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

#TODO do konfigu
data_path = "/home/pepa/projects/camera_filter/learning/diff-%s" % (image_size_x)
model_name = "model-d%s-c%s-1h%s-2h%s" % (image_div, cluster_size, n_hidden_1, n_hidden_2)
model_name = model_name + "-500"

# Construct model
model = MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_classes)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()
    print "opening model %s" % model_name
    saver.restore(sess, "./models/"+model_name)

    p = subprocess.Popen(["find %s -type f | grep 'diff2.png' | sort" % (data_path)], stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    files = output.split()
    positive_count = 0
    negative_count = 0
    positive_mismatch = 0
    negative_mismatch = 0
    print "Prediction mismatches:"
    for filename in files:
        # TODO toto by melo testovat primo obrazek z kamery, nikoliv diff
        npi = pi.read_preprocess_image(filename, cluster_size)
        label = pi.get_image_label(filename)
        npi = npi.reshape(n_input)

        #print "klasifikace: %s, label: %s, image: " % (cl, label, filename)
        cl = sess.run(tf.argmax(model.out_layer, 1), feed_dict={model.input_ph: [npi]})

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
