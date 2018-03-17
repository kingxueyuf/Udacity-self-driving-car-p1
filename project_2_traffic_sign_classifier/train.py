import pickle
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# ---
# ## Step 0: Load The Data
# Load pickled data
training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
n_train = len(X_train)
n_validation = len(X_validation)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = np.max(y_train)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
def normalize_image(images):
    return (images - 128) / 128
# X_train = normalize_image(X_train)
# X_validation = normalize_image(X_validation)
# X_test = normalize_image(X_test)

# ### Model Architecture
def LeNet(x):
    with tf.device('/gpu:0'):
        mu = 0
        sigma = 0.1

        # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean = mu, stddev = sigma), name = 'conv1_w')
        conv1_bias = tf.Variable(tf.zeros(6), dtype=tf.float32, name = 'conv1_bias')
        x = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_bias

        # Activation.
        x = tf.nn.relu(x)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean = mu, stddev = sigma), name = 'conv2_w')
        conv2_bias = tf.Variable(tf.zeros(16),dtype=tf.float32, name = 'conv2_bias')
        x = tf.nn.conv2d(x, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_bias

        # Activation.
        x = tf.nn.relu(x)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1],padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        x = flatten(x)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma), name = 'fc1_w')
        fc1_bias = tf.Variable(tf.zeros(120),dtype=tf.float32, name = 'fc1_bias')
        x = tf.matmul(x, fc1_w) + fc1_bias

        # Activation.
        x = tf.nn.relu(x)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma), name = 'fc2_w')
        fc2_bias = tf.Variable(tf.zeros(84),dtype=tf.float32, name = 'fc2_bias')
        x = tf.matmul(x, fc2_w) + fc2_bias

        # Activation.
        x = tf.nn.relu(x)

        # Layer 5: Fully Connected. Input = 84. Output = 42.
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84,42), mean = mu, stddev = sigma), name = 'fc3_w')
        fc3_bias = tf.Variable(tf.zeros(42),dtype=tf.float32, name = 'fc3_bias')
        x = tf.matmul(x, fc3_w) + fc3_bias

        logits = x

        return logits

# ### Train, Validate and Test the Model
# Hyperparameter
BATCH_SIZE = 200
EPOCHS = 2000
rate = 1e-3

# Placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 42)

# Forward
logits = LeNet(x)
logits_evaluate = LeNet(x)
logits_evaluate = tf.nn.softmax(logits_evaluate)

# Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Backward
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss)

# Evaluate
correct_prediction = tf.equal(tf.argmax(logits_evaluate, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict = {x:batch_x, y:batch_y})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, 'checkpoint/traffic_sign_classifier')
    print("Model saved")


