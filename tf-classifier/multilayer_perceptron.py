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

    # Create model
    def get_model(self, x):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

