#!/usr/bin/env python

import time
import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from lib.config import OptionParser, Configuration
from lib.multilayer_perceptron import MultilayerPerceptron
from lib.preprocess_image import get_image_label
import random
from time import sleep
import sys

#TODO promenne z yml configu

#TODO napad:
# zahrnout informace i z puvodni fotky pred spocitanim rozdilu (napr. podstatne informace
# o jasu cele sceny, tedy noc/den apod..)
# -> to vede na lepsi predzpracovani, vcetne clusterizace
# napr. c/b fotku o stejne velikosti jako clusterovana data
# - dale by slo zahrnout nejakou celkovou informaci (celkova velikost rozdilu)

#TODO sjednotit konstanty do jednoho modulu

# Delitel velikosti obrazku. Kolikrat se obrazek zmensi
# Musi pro to byt predpocitana data
image_div = 2
# Obrazek muzeme rozdelit na shluky cluster_size x cluster_size, u kterych napocitame
# pocet rozdilnych pixelu a na vstup site pujde az toto cislo. V podstate tim zmensime
# pocet vstupu site, bez toho aby se ztratilo tolik informace jako pri prostem zmenseni obrazku
cluster_size = 30

# input image parameters
image_size_x = 1920 / image_div
image_size_y = 1080 / image_div
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
# TODO jaka je optimalni velikost pro danou ulohu a velikost dat?
n_hidden_1 = 200 # 1st layer number of features #256
n_hidden_2 = 100 # 2nd layer number of features  #64
n_input = (image_size_x / cluster_size) * (image_size_y / cluster_size) * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

# Parameters of learning
learning_rate = 0.0001
training_epochs = 5000
batch_size = 50
eval_step = 10
save_step = 0

# Input data
# v data path jsou ocekavany slozky se stejne velkymi obrazky
# slozky zacinajici na t (jako true) jsou brany jako pozitivni klasifikace
# slozky zacinajici na f (jako false) jsou brany jako negativni klasifikace
# Testovaci sadu je vhodne pouzivat pro urceni nejlepsich parametru
# Pro nauceni modelu pro provoz testovaci sadu nepotrebujeme
n_test_pct = 2 # procent testovacich dat

y = tf.placeholder(tf.int64, shape=(None))

def get_files(path):
    print "path = %s" % (path, )
    cmd = "find %s -type f | grep 'c%s.pp' | sort" % (path, cluster_size)
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    print cmd
    (output, err) = p.communicate()
    p_status = p.wait()
    files = output.split()

    train_files = []
    test_files = []
    # jako testovaci bereme kazdy x-ty obrazek (x = int(100 / n_test_pct))
    test_counter = int(100 / n_test_pct) if n_test_pct else 0
    for file in files:
        test_counter -= 1
        if not test_counter:
            test_files.append(file)
            test_counter = int(100 / n_test_pct)
        else:
            train_files.append(file)

    print "Num files = %s" % len(files)
    print "Train images = %s" % len(train_files)
    print "Test images = %s" % len(test_files)
    return train_files, test_files


def get_images_labels(files):
    images = []
    labels = []
    for filename in files:
        #image = pi.read_preprocess_image(filename, cluster_size)
        f = open(filename, "rb")
        image = np.fromfile(f, dtype=np.uint16)
        f.close()
        label = get_image_label(filename)
        images.append(image)
        labels.append(label)
    np_images = np.array(images)
    np_images = np_images.reshape(len(images), n_input)
    np_labels = np.array(labels)
    return np_images, np_labels

def main(argv):
    parser = OptionParser()
    options, args = parser.parse_args_dict()

    cfg = Configuration(options)

    learn_dir = cfg.yaml['main']['learn_dir']
    model_dir = cfg.yaml['main']['model_dir']
    data_dir = "%s/diff-%s" % (learn_dir, image_size_x)

    model_name = "model-d%s-c%s-1h%s-2h%s" % (image_div, cluster_size, n_hidden_1, n_hidden_2)
    print model_name

    # Construct model
    model = MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_classes)

    # Define loss function
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.out_layer, labels=y)) #label je index spravneho vystupu
    # Define optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    #TODO vyzkouset jine optimalizace
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Test model
    correct_prediction = tf.equal(tf.argmax(model.out_layer, 1), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        train_files, test_files = get_files(data_dir)
        n_train = len(train_files)
        print "reading testing data"
        start = time.time()
        test_images, test_labels = get_images_labels(test_files)
        end = time.time()
        print "testing data complete (%s s)" % (end - start)

        print "reading training data"
        start = time.time()
        train_images, train_labels = get_images_labels(train_files)
        end = time.time()
        print "training data complete (%s s)" % (end - start)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            index_in_epoch = 0
            total_batches = int(n_train/batch_size)

            # nahodne zamichame obrazky i labely ve stejnem poradi
            # random.random nefunguje spravne na vicerozmerne np.array
            # TODO je to k necemu?
            r = np.arange(len(train_images))
            np.random.shuffle(r)
            train_images = train_images[r]
            train_labels = train_labels[r]

            # pokud data nejsou delitelna davkou, chcem pouzit i zbytek
            if n_train > total_batches * batch_size:
                total_batches += 1
            # Loop over all batches
            for i in range(total_batches):
                _, c = sess.run([optimizer, cost], feed_dict={model.input_ph: train_images[index_in_epoch : index_in_epoch+batch_size],
                                                              y: train_labels[index_in_epoch : index_in_epoch+batch_size]})
                # Compute average loss
                avg_cost += c / total_batches
                index_in_epoch += batch_size

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch), "cost=", \
                "{:.9f}".format(avg_cost))
            if epoch % eval_step == 0:
                train_acc = accuracy.eval({model.input_ph: train_images, y: train_labels})
                print("Accuracy on train images:", train_acc)
                if n_test_pct:
                    test_acc = accuracy.eval({model.input_ph: test_images, y: test_labels})
                    print("Accuracy on test images:", test_acc)
                    if test_acc == 1.0 or avg_cost < 1.0:
                        break
                elif train_acc == 1.0:
                    break
            if epoch and save_step and epoch % save_step == 0:
                saver.save(sess, "%s/%s" % (learn_dir, model_name), global_step=epoch)
            # toto je tu zamerne, kvuli snizeni vytizeni procesoru
            #sleep(0.5)
        print("Optimization Finished!")

        saver.save(sess, "%s/%s" % (model_dir, model_name))

        # TODO presunout do testeru
        print "prediction mismatches in test data:"
        dt = 0
        df = 0
        mt = 0
        mf = 0
        for n, test_image in enumerate(test_images):
            cl = sess.run(tf.argmax(model.out_layer, 1), feed_dict={model.input_ph: [test_image]})
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
        print "Model %s" % model_name
        train_acc = accuracy.eval({model.input_ph: train_images, y: train_labels})
        test_acc = accuracy.eval({model.input_ph: test_images, y: test_labels})
        print("Accuracy on train images:", train_acc)
        print("Accuracy on test images:", test_acc)
        print "%s/%s true and %s/%s false mismatches on test images" % (mt, dt, mf, df)

if __name__ == "__main__":
    main(sys.argv[1:])
