#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
import multilayer_perceptron as mp
from random import shuffle
from time import sleep
import sys

#TODO napad:
# dalo by se rozdelit obrazek na sekce a v tech spocitat pocet rozdilnych pixelu.
# Zmensilo by to pocet vstupu, ale neztratilo by se tolik informace, jako pri prostem zmenseni!
# Byl by to takovy lepsi resize

# Parameters
learning_rate = 0.0001
training_epochs = 10000
batch_size = 50
eval_step = 10
save_step = 1000

# zatim nejlepsi vysledky mi dava FUZZ=10, obrazek/10, sit 256x32
# input image parameters
FUZZ = 10
img_resize = 10
image_size_x = 1920 / img_resize
image_size_y = 1080 / img_resize
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
# TODO jaka je optimalni velikost pro danou ulohu a velikost dat?
n_hidden_1 = 64 # 1st layer number of features #24
n_hidden_2 = 32 # 2nd layer number of features  #16
n_input = image_size_x * image_size_y * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)

# Input data
# v data path jsou ocekavany slozky se stejne velkymi obrazky
# slozky zacinajici na t (jako true) jsou brany jako pozitivni klasifikace
# slozky zacinajici na f (jako false) jsou brany jako negativni klasifikace
data_path = "/home/pepa/projects/camera_filter/learning/diff%s-%s" % (FUZZ, image_size_x)
n_test_pct = 10 # procent testovacich dat

model_name = "model-%s-%s-%s-%s" % (FUZZ, image_resize, n_hidden_1, n_hidden_2)

x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.int64, shape=(None))

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
            #TODO pravdepodobnost pro true false?
            labels.append(0) #negative alarm
            #print ("%s (0)" % (filename, ))
        else:
            labels.append(1) #positive alarm (ma vyssi vahu?)
            #print ("%s (1)" % (filename, ))
        image = Image.open(filename)
        # TODO if image.size <> network size?
        #image = image.resize((image_size_x, image_size_y))
        images.append(np.array(image))
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
            print("Epoch:", '%04d' % (epoch), "cost=", \
                "{:.9f}".format(avg_cost))
            if epoch % eval_step == 0:
                train_acc = accuracy.eval({x: train_images, y: train_labels})
                test_acc = accuracy.eval({x: test_images, y: test_labels})
                print("Accuracy on train images:", train_acc)
                print("Accuracy on test images:", test_acc)
                if train_acc > best_acc:
                    best_acc = train_acc
                    saver.save(sess, "./" + model_name, global_step=epoch)
            if epoch % save_step == 0:
                saver.save(sess, "./" + model_name, global_step=epoch)
            # toto je tu zamerne, kvuli snizeni vytizeni procesoru
            sleep(1)
        print("Optimization Finished!")

        saver.save(sess, "./" + model_name)

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

        train_images, train_labels = get_images_labels(train_files)
        print("Accuracy on train images:", accuracy.eval({x: train_images, y: train_labels}))
        print("Accuracy on test images:", accuracy.eval({x: test_images, y: test_labels}))

if __name__ == "__main__":
    main(sys.argv[1:])
