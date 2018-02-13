import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


# define global variable
img_size = 32
num_channels = 3
num_classes = 10
num_file_train = 5
img_per_file = 10000
num_images_train = num_file_train * img_per_file
data_path = 'cifar-10/'


def get_file_path(filename=""):
    return os.path.join(data_path, filename)


def unpickle(filename):
    file_path = get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data


def convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images


def load_data(filename):
    data = unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = convert_images(raw_images)
    return images, cls


def load_class_names():
    raw = unpickle(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names


def load_training_data():
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images_train], dtype=int)
    begin = 0
    for j in range(num_file_train):
        images_batch, cls_batch = load_data(filename="data_batch_" + str(j + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    return images, cls, lb.transform(cls)


def load_test_data():
    images, cls = load_data(filename="test_batch")
    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    return images, cls, lb.transform(cls)


if __name__ == '__main__':
    train_images, train_cls, train_label = load_training_data()
    test_images, test_cls, test_label = load_test_data()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    K = 16
    L = 16
    M = 16

    W1 = tf.Variable(tf.truncated_normal([3, 3, 3, K], stddev=0.1))
    b1 = tf.Variable(tf.ones([K]) / 10)

    W2 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=0.1))
    b2 = tf.Variable(tf.ones([L]) / 10)

    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    b3 = tf.Variable(tf.ones([M]) / 10)

    N1 = 384
    N2 = 192

    WW1 = tf.Variable(tf.truncated_normal([8 * 8 * M, N2], stddev=0.1))
    bb1 = tf.Variable(tf.ones([N2]) / 10)

    WW2 = tf.Variable(tf.truncated_normal([N1, N2], stddev=0.1))
    bb2 = tf.Variable(tf.ones([N2]) / 10)

    WW3 = tf.Variable(tf.truncated_normal([N2, 10], stddev=0.1))
    bb3 = tf.Variable(tf.ones([10]) / 10)

    y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    pool1 = tf.nn.max_pool(y1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # batch1 = tf.layers.batch_normalization(pool1, center=True, scale=True, training=True, fused=True)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=1e-4, beta=0.75)

    y2 = tf.nn.relu(tf.nn.conv2d(norm1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
    norm2 = tf.nn.lrn(y2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # batch2 = tf.layers.batch_normalization(pool2, center=True, scale=True, training=True, fused=True)

    # y3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)
    # pool3 = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # batch3 = tf.layers.batch_normalization(pool3, center=True, scale=True, training=True, fused=True)
    # norm3 = tf.nn.lrn(batch3, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)

    yy = tf.reshape(pool2, shape=[-1, 8 * 8 * M])

    yy1 = tf.nn.relu(tf.matmul(yy, WW1) + bb1)
    bbatch1 = tf.layers.batch_normalization(yy1, center=True, scale=True, training=True, fused=True)
    # yy2 = tf.nn.relu(tf.matmul(bbatch1, WW2) + bb2)
    # bbatch2 = tf.layers.batch_normalization(yy2, center=True, scale=True, training=True, fused=True)
    y = tf.nn.softmax(tf.matmul(bbatch1, WW3) + bb3)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = tf.reduce_mean(cross_entropy)

    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 0.01
    # lr = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.95, staircase=True)

    lr =9e-4
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    loop = 10000
    n = 128

    # for j in range(train_images.shape[0]):
    #     if j % 1000 == 0:
    #         print(j)
    #
    #     img = tf.random_crop(train_images[j], size=[32, 32, 3])
    #     img = tf.image.random_flip_left_right(img)
    #     img = tf.image.random_hue(img, max_delta=0.05)
    #     img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
    #     img = tf.image.random_brightness(img, max_delta=0.2)
    #     train_images[j] = tf.image.random_saturation(img, lower=0.0, upper=2.0).eval()

    for i in range(loop):
        index = [j % num_images_train for j in range(n * i, n * (i + 1))]
        batch_x = train_images[index]
        batch_y = train_label[index]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_batch = []

    for i in range(20):
        test_images_batch = test_images[(i*500):(500*i+500)]
        test_label_batch = test_label[(i * 500):(500 * i + 500)]
        test_batch.append(accuracy.eval(feed_dict={x: test_images_batch, y_: test_label_batch}))

    print('Test data accuracy: {}'.format(np.mean(test_batch)))
    sess.close()
