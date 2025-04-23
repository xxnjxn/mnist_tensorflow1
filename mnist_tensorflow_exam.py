
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd

# Model Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 100

# Load MNIST dataset using Keras datasets API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Network Parameters
n_input = 784  # MNIST data input (28*28 pixels)
n_hidden1 = 512
n_hidden2 = 256
n_classes = 10  # MNIST total classes (0-9 digits)

# Placeholders for input and output
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

# Weights and biases initialization
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.global_variables_initializer()
total_batch = x_train.shape[0] // batch_size

# Create a Dataset object for batching & batch size iterator generation
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
iterator = tf.data.make_initializable_iterator(dataset)
next_batch = iterator.get_next()

# Start Training
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)

    for epoch in range(num_epochs):
        avg_loss = 0.
        for i in range(total_batch):
            batch_x, batch_y = sess.run(next_batch)
            _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += l / total_batch

        acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
        print("Epoch:", '%02d' % (epoch+1), "Loss:", "{:.4f}".format(avg_loss), "Accuracy:", "{:.4f}".format(acc) + "%")
        sess.run(iterator.initializer)

    final_acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("Final Test Accuracy:", "{:.4f}".format(final_acc) + "%")
    print()
    print("Tensorflow:", tf.__version__)
    print()

    data = {
        '이름': ['이은진'],
        '학번': [2572307],
        '학과': ['산업디자인학과']
    }

    df = pd.DataFrame(data)
    print(df)
    print()
