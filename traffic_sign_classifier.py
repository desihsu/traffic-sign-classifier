import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def preprocess_data(X_train, X_valid, X_test):
    # Convert to grayscale
    X_train = np.mean(X_train, axis=3, keepdims=True)
    X_valid = np.mean(X_valid, axis=3, keepdims=True)
    X_test = np.mean(X_test, axis=3, keepdims=True)

    # Normalize
    X_train = (X_train - 128) / 128
    X_valid = (X_valid - 128) / 128
    X_test = (X_test - 128) / 128

    return X_train, X_valid, X_test


# Model architecture
def inception(x):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    w1 = tf.Variable(tf.truncated_normal(shape=(5,5,1,32), 
                                         mean=mu, stddev=sigma), name='w1')
    b1 = tf.Variable(tf.zeros(32), name='b1')
    layer1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') + b1
    
    # Activation & Max pool. Output = 14x14x32.
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], 
                            padding='VALID')
    
    # Layer 2: Convolutional. Input = 14x14x32. Output = 10x10x64.
    w2 = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), 
                                         mean=mu, stddev=sigma), name='w2')
    b2 = tf.Variable(tf.zeros(64), name='b2')
    layer2 = tf.nn.conv2d(layer1, w2, strides=[1,1,1,1], padding='VALID') + b2
    
    # Activation & Max pool. Output = 5x5x64.
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], 
                            padding='VALID')
    
    # Layer 3: Convolutional. Input = 5x5x64. Output = 3x3x128.
    w3 = tf.Variable(tf.truncated_normal(shape=(3,3,64,128), 
                                         mean=mu, stddev=sigma), name='w3')
    b3 = tf.Variable(tf.zeros(128), name='b3')
    layer3 = tf.nn.conv2d(layer2, w3, strides=[1,1,1,1], padding='VALID') + b3
    
    # Activation & Max pool. Output = 2x2x128.
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.max_pool(layer3, ksize=[1,2,2,1], strides=[1,1,1,1], 
                            padding='VALID')
    
    # Max pool, Flatten, Concat, & Dropout. Output = 1920
    layer1 = tf.nn.max_pool(layer1, ksize=[1,4,4,1], strides=[1,2,2,1], 
                            padding='VALID')
    layer2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], 
                            padding='VALID')
    fc0 = tf.concat([flatten(layer1), flatten(layer2), flatten(layer3)], 1)
    fc0 = tf.nn.dropout(fc0, keep_prob=0.5)
    
    # Layer 4: Fully connected layer. Input = 1920. Output = 800.
    w4 = tf.Variable(tf.truncated_normal(shape=(1920,800), 
                                         mean=mu, stddev=sigma), name='w4')
    b4 = tf.Variable(tf.zeros(800), name='b4')
    layer4 = tf.matmul(fc0, w4) + b4
    layer4 = tf.nn.relu(layer4)
    
    # Layer 5: Fully connected layer. Input = 800. Output = 43.
    w5 = tf.Variable(tf.truncated_normal(shape=(800,43), 
                                         mean=mu, stddev=sigma), name='w5')
    b5 = tf.Variable(tf.zeros(43), name='b5')
    logits = tf.matmul(layer4, w5) + b5
    
    return logits


def train_model(X_train, y_train, X_valid, y_valid):
    n_classes = len(np.unique(y_train))
    n_examples = len(X_train)

    EPOCHS = 20
    BATCH_SIZE = 128
    rate = 0.0005

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    keep_prob = tf.placeholder(tf.float32)

    logits = inception(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,
                                                            logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = (X_data[offset:offset+BATCH_SIZE], 
                                y_data[offset:offset+BATCH_SIZE])
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, 
                                                               y: batch_y, 
                                                               keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training...\n")

        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)

            for offset in range(0, n_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,
                                                        keep_prob: 0.5})

            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

        saver.save(sess, "./model")
        print("Model saved")

        train_accuracy = evaluate(X_train, y_train)
        valid_accuracy = evaluate(X_valid, y_valid)
        test_accuracy = evaluate(X_test, y_test)
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(valid_accuracy))
        print("Test Accuracy = {:.3f}".format(test_accuracy))


if __name__ == "__main__":
    training_file = 'traffic-signs-data/train.p'
    validation_file = 'traffic-signs-data/valid.p'
    testing_file = 'traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test)
    train_model(X_train, y_train, X_valid, y_valid)