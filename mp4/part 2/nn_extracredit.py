from __future__ import  print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import time

# data loader.

x_train = np.load("data/x_train.npy")
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
y_train = np.load("data/y_train.npy")

x_test = np.load("data/x_test.npy")
x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
y_test = np.load("data/y_test.npy")

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Reformat into a shape that's more adapted to the models we're going to train:
# data as a flat matrix,
# labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_channels = 1  # gray scale = (R+G+B)/3


# adapt reformat function to 2d-conv
# turn label into 1-hot encoding
def reformat(dataset, labels):
    reformatted_data = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    reformatted_label = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return reformatted_data, reformatted_label

# @TODO add extract 500 data from test set as validation set

train_dataset, train_label = reformat(x_train, y_train)
test_dataset, test_label = reformat(x_test, y_test)
#valid_dataset, valid_label = reformat(valid_dataset, valid_labels)


print(train_dataset.shape, train_label.shape)
print(test_dataset.shape, test_label.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# building graph
batch_size = 32
patch_size = 8
depth = 32
hidden = 128

graph_conv = tf.Graph()

with graph_conv.as_default():


    train_data = tf.placeholder(tf.float32, [batch_size, image_size, image_size, num_channels])
    train_labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    #valid_data = tf.constant(valid_dataset)
    test_data = tf.constant(test_dataset)

    conv_w_1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    conv_b_1 = tf.Variable(tf.truncated_normal([depth]))

    conv_w_2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    conv_b_2 = tf.Variable(tf.constant(1.0, shape=[depth]))

    conv_w_3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden], stddev=0.1))
    conv_b_3 = tf.Variable(tf.constant(1.0, shape=[hidden]))

    conv_w_4 = tf.Variable(tf.truncated_normal([hidden, num_labels], stddev=0.1))
    conv_b_4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    def conv_model(data):
        conv = tf.nn.conv2d(data, conv_w_1, [1, 2, 2, 1], padding="SAME", use_cudnn_on_gpu=True)
        hidden_l = tf.nn.relu(conv + conv_b_1)
        conv = tf.nn.conv2d(hidden_l, conv_w_2, [1, 2, 2, 1], padding="SAME", use_cudnn_on_gpu=True)
        hidden_l = tf.nn.relu(conv + conv_b_2)
        shape = hidden_l.get_shape().as_list()
        reshape = tf.reshape(hidden_l, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden_l = tf.nn.relu(tf.matmul(reshape, conv_w_3) + conv_b_3)
        return tf.matmul(hidden_l, conv_w_4) + conv_b_4


    logits = conv_model(train_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(conv_model(test_data))
    #valid_prediction = tf.nn.softmax(conv_model(valid_data))

num_steps = 1001

with tf.Session(graph=graph_conv, config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    start_time = time.time()
    print("Variables initialized")
    for step in range(num_steps):
        offset = (step*batch_size) % (train_label.shape[0]-batch_size)
        batch_data = train_dataset[offset:(offset+batch_size), :, :, :]
        batch_label = train_label[offset:(offset+batch_size), :]
        feed_dict = {train_data: batch_data, train_labels: batch_label}
        _, l, prediction = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print("At step %d, the training loss is %.2f" % (step, l))
            print("The training prediction is %.2f%%" % accuracy(prediction, batch_label))
            #print("The validation prediction is %.2f%%" % accuracy(valid_prediction.eval(), valid_label))
    end_time = time.time()
    print("Test prediction is %.2f%%" % accuracy(test_prediction.eval(), test_label))
    print("Total cost %.2fs" % (end_time - start_time))
