#!/usr/bin/env python

import time
import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import multilayer_perceptron as mp
import preprocess_image as pi
from random import shuffle
from time import sleep
import sys

#TODO napad:
# zahrnout informace i z puvodni fotky pred spocitanim rozdilu (napr. podstatne informace
# o jasu cele sceny, tedy noc/den apod..)
# -> to vede na lepsi predzpracovani, vcetne clusterizace
# napr. c/b fotku o stejne velikosti jako clusterovana data

# Parameters
learning_rate = 0.0001
training_epochs = 200
batch_size = 50
eval_step = 10
save_step = 500

# Nejednoznacnost pri urcovani rozdilu v obrazku. Cim vyssi cislo, tim mene rozdilu
# Musi pro to byt predpocitana data
FUZZ = 12
# Delitel velikosti obrazku. Kolikrat se obrazek zmensi
# Musi pro to byt predpocitana data
image_div = 2
# Obrazek muzeme rozdelit na shluky cluster_size x cluster_size, u kterych napocitame
# pocet rozdilnych pixelu a na vstup site pujde az toto cislo. V podstate tim zmensime
# pocet vstupu site, bez toho aby se ztratilo tolik informace jako pri prostem zmenseni obrazku
cluster_size = 10

# input image parameters
image_size_x = 1920 / image_div
image_size_y = 1080 / image_div
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
# TODO jaka je optimalni velikost pro danou ulohu a velikost dat?
n_hidden_1 = 128 # 1st layer number of features #256
n_hidden_2 = 64 # 2nd layer number of features  #64
n_input = (image_size_x / cluster_size) * (image_size_y / cluster_size) * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

# Input data
# v data path jsou ocekavany slozky se stejne velkymi obrazky
# slozky zacinajici na t (jako true) jsou brany jako pozitivni klasifikace
# slozky zacinajici na f (jako false) jsou brany jako negativni klasifikace
data_path = "/home/pepa/projects/camera_filter/learning/diff-f%s-%s" % (FUZZ, image_size_x)
n_test_pct = 25 # procent testovacich dat

model_name = "model-%s-%s-%s-%s-%s" % (FUZZ, image_div, cluster_size, n_hidden_1, n_hidden_2)

x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.int64, shape=(None))

def get_files(path):
    print "path = %s" % (path, )
    p = subprocess.Popen(["find %s -type f | grep '.pp' | sort -R" % (path)], stdout=subprocess.PIPE, shell=True)
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
        #image = pi.read_preprocess_image(filename, cluster_size)
        f = open(filename, "rb")
        image = np.fromfile(f, dtype=np.int16)
        f.close()
        label = pi.get_image_label(filename)
        images.append(image)
        labels.append(label)
    np_images = np.array(images)
    np_images = np_images.reshape(len(images), n_input)
    np_labels = np.array(labels)
    return np_images, np_labels

def main(argv):

    # Construct model
    multilayer_perceptron = mp.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_classes)
    pred = multilayer_perceptron.get_model(x)

    # Define loss function
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)) #label je index spravneho vystupu
    # Define optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    #TODO vyzkouset jine optimalizace
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        train_files, test_files = get_files(data_path)
        n_train = len(train_files)
        print "reading testing data"
        start = time.time()
        test_images, test_labels = get_images_labels(test_files)
        end = time.time()
        print "testing data complete (%s s)" % (end - start)

        # TODO zamichat spolu se stitky pro kazdou epochu
        shuffle(train_files)
        print "reading training data"
        start = time.time()
        train_images, train_labels = get_images_labels(train_files)
        end = time.time()
        print "training data complete (%s s)" % (end - start)

        best_acc = 0
        # Training cycle
        for epoch in range(training_epochs):
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
            print("Epoch:", '%04d' % (epoch), "cost=", \
                "{:.9f}".format(avg_cost))
            if epoch % eval_step == 0:
                train_acc = accuracy.eval({x: train_images, y: train_labels})
                test_acc = accuracy.eval({x: test_images, y: test_labels})
                print("Accuracy on train images:", train_acc)
                print("Accuracy on test images:", test_acc)
                if (train_acc + test_acc) > best_acc:
                    best_acc = train_acc + test_acc
                    saver.save(sess, "./" + model_name, global_step=epoch)
            if epoch % save_step == 0:
                saver.save(sess, "./" + model_name, global_step=epoch)
            # toto je tu zamerne, kvuli snizeni vytizeni procesoru
            sleep(2)
        print("Optimization Finished!")

        saver.save(sess, "./" + model_name)

        # TODO presunout do testeru
        print "Prediction mismatches in test data:"
        dt = 0
        df = 0
        mt = 0
        mf = 0
        for n, test_image in enumerate(test_images):
            cl = sess.run(tf.argmax(pred, 1), feed_dict={x: [test_image]})
            label = test_labels[n]
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

if __name__ == "__main__":
    main(sys.argv[1:])
