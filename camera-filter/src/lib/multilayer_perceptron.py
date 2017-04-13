import tensorflow as tf

class MultilayerPerceptron:
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_classes):
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_classes = n_classes
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.input_ph = tf.placeholder(tf.float32, shape=(None, n_input))

        # Create model
        # Hidden layer with RELU activation
        self.layer_1 = tf.add(tf.matmul(self.input_ph, self.weights['h1']), self.biases['b1'])
        self.layer_1 = tf.nn.relu(self.layer_1)
        # Hidden layer with RELU activation
        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2'])
        self.layer_2 = tf.nn.relu(self.layer_2)
        # Output layer with linear activation
        self.out_layer = tf.matmul(self.layer_2, self.weights['out']) + self.biases['out']
