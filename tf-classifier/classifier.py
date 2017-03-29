#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import multilayer_perceptron as mp
from random import shuffle
from time import sleep

# Parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 50
eval_step = 10
save_step = 100

# zatim nejlepsi vysledky mi dava obrazek/10, FUZZ 10, sit 1024x32
# input image parameters
FUZZ = 10
image_size_x = 1920 / 10
image_size_y = 1080 / 10
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
# TODO jaka je optimalni velikost pro danou ulohu a velikost dat?
n_hidden_1 = 256 # 1st layer number of features #24
n_hidden_2 = 16 # 2nd layer number of features  #16
n_input = image_size_x * image_size_y * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

# Input data
# v data path jsou ocekavany slozky se stejne velkymi obrazky
# slozky zacinajici na t (jako true) jsou brany jako pozitivni klasifikace
# slozky zacinajici na f (jako false) jsou brany jako negativni klasifikace
data_path = "/home/pepa/projects/camera_filter/learning/diff%s-%s" % (FUZZ, image_size_x)
n_test_pct = 10 # procent testovacich dat


def get_files(path):
    print "path = %s" % (path, )
    p = subprocess.Popen(["find %s -type f | sort -R" % (path)], stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    files = output.split()
    n_files = len(files)
    n_test = int(n_files / 100) * n_test_pct
    n_train = n_files - n_test
    train_files = files[:n_train]
    test_files = files[n_train:]
    print "Num files = %s" % (n_files, )
    print "Train images = %s" % (n_train, )
    print "Test images = %s" % (n_test, )
    return train_files, test_files

def get_images_labels(files):
    images = []
    labels = []
    for filename in files:
        # podle prviho pismena posledni slozky souboru pozname true/false
        if filename.split("/")[-2:][0][0] == 'f': #false
            #TODO vahy pro true false?
            labels.append(np.array((1, 0))) #negative alarm
            #print ("%s (0)" % (filename, ))
        else:
            labels.append(np.array((0, 2))) #positive alarm (ma vyssi vahu?)
            #print ("%s (1)" % (filename, ))
        image = Image.open(filename)
        # TODO if image.size <> network size?
        #image = image.resize((image_size_x, image_size_y))
        images.append(np.array(image))
    np_images = np.array(images)
    np_images = np_images.reshape(len(images), n_input)
    np_labels = np.array(labels)
    return np_images, np_labels

# nacist data
#   p = subprocess.Popen(["find %s -type f | sort -R" % (data_path)], stdout=subprocess.PIPE, shell=True)
#   (output, err) = p.communicate()
#   p_status = p.wait()
#   files = output.split()
#   input_images = []
#   input_labels = []
#   n_examples = len(files)
#   n_test = int(n_examples / 100) * n_test_pct
#   n_train = n_examples - n_test
#   for filename in files:
#     # TODO udelat lepe rozpoznani true/false
#     # /home/pepa/projects/tensorflow/pokus1/data/true/ARC20170318095401-diff.pn
#     if filename[43] == 'f': #false
#TODO vahy true false?
#       input_labels.append(np.array((1, 0))) #negative alarm
#       #print ("%s (0)" % (filename, ))
#     else:
#       input_labels.append(np.array((0, 1))) #positive alarm
#       #print ("%s (1)" % (filename, ))
#     image = Image.open(filename)
#     #image = image.resize((image_size_x, image_size_y))
#     input_images.append(np.array(image))
# TODO rozdelit vstupni data na trenovaci a testovaci mnozinu
# TODO nahodne zamichat pro kazdou epochu
#   train_images = np.array(input_images[:n_train])
#   train_images = train_images.reshape(n_train, n_input)
#   train_labels = np.array(input_labels[:n_train])
#   test_images = np.array(input_images[n_train:])
#   test_images = test_images.reshape(n_test, n_input)
#   test_labels = np.array(input_labels[n_train:])
#train_labels = train_labels.reshape(n_train, n_classes) # neni treba, jiz je spravne

# tf Graph input
# FIXME float nebo int?
x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes))


# Construct model
multilayer_perceptron = mp.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_classes)
pred = multilayer_perceptron.get_model(x)

#TODO vyzkouset jine optimalizace
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#labels = tf.to_int64(labels)
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#  logits=pred, labels=y, name='xentropy')
#cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
#global_step = tf.Variable(0, name='global_step', trainable=False)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #correct_prediction = tf.nn.in_top_k(tf.argmax(pred, 1), tf.argmax(y, 1), 1)
    #accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    train_files, test_files = get_files(data_path)
    n_train = len(train_files)
    test_images, test_labels = get_images_labels(test_files)

    best_acc = 0
    # Training cycle
    for epoch in range(training_epochs):
        shuffle(train_files)
        # TODO nejak se mi nelibi ze pro kazdou epochu musim znovu nacitat obrazky (ale zase to setri pamet)
        # je to tu hlavne kvuli moznosti zamichat testovaci data pro kazdou epochu
        train_images, train_labels = get_images_labels(train_files)
        avg_cost = 0.
        index_in_epoch = 0
        total_batches = int(n_train/batch_size)
        # pokud data nejsou delitelna davkou, chcem pouzit i zbytek
        if n_train > total_batches * batch_size:
            total_batches += 1
        # Loop over all batches
        for i in range(total_batches):
            _, c = sess.run([optimizer, cost], feed_dict={x: train_images[index_in_epoch : index_in_epoch+batch_size],
                                                          y: train_labels[index_in_epoch : index_in_epoch+batch_size]})
            # Compute average loss
            avg_cost += c / total_batches
            index_in_epoch += batch_size

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
        if epoch % eval_step == 0:
            train_acc = accuracy.eval({x: train_images, y: train_labels})
            test_acc = accuracy.eval({x: test_images, y: test_labels})
            print("Accuracy on train images:", train_acc)
            print("Accuracy on test images:", test_acc)
            if train_acc > best_acc:
                best_acc = train_acc
                saver.save(sess, 'model', global_step=epoch)
        if epoch % save_step == 0:
            saver.save(sess, 'model', global_step=epoch)
        # TODO toto je tu kvuli snizeni vytizeni procesoru
        sleep(1)
    print("Optimization Finished!")

    # TODO ukladat do nazvu modelu parametry
    saver.save(sess, 'model')

    # TODO zkouska na konkretnim obrazku
    #cl = pred.eval(feed_dict={x: [mnist.test.images[0]]})
    print "Prediction mismatches in test data:"
    dt = 0
    df = 0
    mt = 0
    mf = 0
    for n, test_image in enumerate(test_images):
        cl = sess.run(tf.argmax(pred, 1), feed_dict={x: [test_image]})
        # tf.argmax(pred, 1) vrati index vyssiho cisla z pole vystupnich neuronu, tedy 0 pro false alarm a 1 pro true alarm
        # v test_labels[n] je (True, False) pro false alarm a (False, True) pro true alarm
        label = 1 if test_labels[n][1] else 0
        #print cl, test_labels[n], label
        if label:
            dt = dt + 1
        else:
            df = df + 1
        if cl != label:
            print "Prediction: %s, label: %s, test file: %s" % (cl, label, test_files[n])
            if label:
                mt = mt + 1
            else:
                mf = mf + 1
    print "%s/%s true and %s/%s false mismatches" % (mt, dt, mf, df)

    train_images, train_labels = get_images_labels(train_files)
    print("Accuracy on train images:", accuracy.eval({x: train_images, y: train_labels}))
    print("Accuracy on test images:", accuracy.eval({x: test_images, y: test_labels}))

