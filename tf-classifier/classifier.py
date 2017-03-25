#!/usr/bin/env python

import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image

# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 50
display_step = 1

# input image parameters
image_size_x = 960 #480
image_size_y = 540 #270
channels = 1 # R,G,B = 3 B/W = 1

# Network Parameters
n_hidden_1 = 16 # 1st layer number of features #24
n_hidden_2 = 8 # 2nd layer number of features  #16
n_input = image_size_x * image_size_y * channels # MNIST data input
n_classes = 2 # MNIST total classes (negative alarm, positive alarm) (pocet vystupu ze site)
n_test_pct = 50 # procent testovacich dat

# Input data
# v data path jsou ocekavany slozky se stejne velkymi obrazky
# slozky zacinajici na t (jako true) jsou brany jako pozitivni klasifikace
# slozky zacinajici na f (jako false) jsou brany jako negativni klasifikace
data_path = "~/projects/tensorflow/pokus1/data"

# nacist data
input_images = []
input_labels = []
p = subprocess.Popen(["find %s -type f | sort -R" % (data_path)], stdout=subprocess.PIPE, shell=True)
(output, err) = p.communicate()
p_status = p.wait()
files = output.split()
n_examples = len(files)
n_test = int(n_examples / 100) * n_test_pct
n_train = n_examples - n_test
for filename in files:
  # TODO udelat lepe rozpoznani true/false
  # /home/pepa/projects/tensorflow/pokus1/data/true/ARC20170318095401-diff.pn
  if filename[43] == 'f': #false
    input_labels.append(np.array((1, 0))) #negative alarm
    print ("%s (0)" % (filename, ))
  else:
    input_labels.append(np.array((0, 1))) #positive alarm
    print ("%s (1)" % (filename, ))
  image = Image.open(filename)
  #image = image.resize((image_size_x, image_size_y))
  input_images.append(np.array(image))
# TODO rozdelit vstupni data na trenovaci a testovaci mnozinu
# TODO nahodne zamichat pro kazdou epochu
train_images = np.array(input_images[0:n_train])
train_images = train_images.reshape(n_train, n_input)
train_labels = np.array(input_labels[0:n_train])
test_images = np.array(input_images[0:n_test])
test_images = test_images.reshape(n_test, n_input)
test_labels = np.array(input_labels[0:n_test])
#train_labels = train_labels.reshape(n_train, n_classes) # neni treba, jiz je spravne
print ("Num examples: %s", n_examples)
print ("train images: %s", n_train)
print ("test images: %s", n_test)

# tf Graph input
# FIXME float nebo int?
x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#TODO vyzkouset jine optimalizace
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
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #correct_prediction = tf.nn.in_top_k(tf.argmax(pred, 1), tf.argmax(y, 1), 1)
    #accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    # TODO testovaci davka misto trenovaci
    print("Accuracy on train images:", accuracy.eval({x: train_images, y: train_labels}))
    print("Accuracy on test images:", accuracy.eval({x: test_images, y: test_labels}))

    # TODO zkouska na konkretnim obrazku
    #cl = pred.eval(feed_dict={x: [mnist.test.images[0]]})
    #for n, train_image in enumerate(train_images):
    #    cl = sess.run(tf.argmax(pred, 1), feed_dict={x: [train_image]})
    #    print (cl, train_labels[n])

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    saver.save(sess, 'model', global_step=training_epochs)

