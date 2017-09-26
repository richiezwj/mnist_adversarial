import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.backends.backend_pdf import PdfPages
import deepnn


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn.deepnn(x)

    # training hyper-parameters
    learning_rate = 0.005

    with tf.name_scope('prediction'):
        # calculate the probability of each class using an extra softmax layer with the final prediction class
        y_prob = tf.nn.softmax(y_conv)
        prediction = tf.argmax(y_prob, 1)

    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    x_delta = tf.gradients(loss, x)

    with tf.name_scope('sign_gradient_update'):
        sign_gradient_x = tf.stop_gradient(x - tf.sign(x_delta) * learning_rate)
        sign_gradient_x = tf.clip_by_value(sign_gradient_x, 0, 1)

    true_label = FLAGS.true_class
    target_label = FLAGS.target_class

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Loaded trained model
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.saved_model)

        # Choose images labeled as "true class"
        [probs, preds] = sess.run([y_prob, prediction], feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        images_indices = [i for i, y in enumerate(preds) if y == true_label]
        chosen_images_probs = [prob[true_label] for i, prob in enumerate(probs) if i in images_indices]
        chosen_images = [image for i, image in enumerate(mnist.test.images) if i in images_indices][0:FLAGS.num_images]

        np.save("images/chosen_images.npy", chosen_images)
        np.save("images/chosen_images_probs.npy", chosen_images_probs)

        fake_label = np.zeros(10)
        fake_label[target_label] = 1

        pp = PdfPages("images/iterate_sign_gradient.pdf")

        # Create adversarial image one by one using iterative sign gradient method
        # inspired by: http://slazebni.cs.illinois.edu/spring17/lec10_visualization.pdf
        for idx, image in enumerate(chosen_images):
            x_updated = image
            result = true_label
            distort_prob = 0
            delta = image
            while result != target_label:
                x_reshape = np.reshape(x_updated, [-1, 784])
                delta, x_updated = sess.run([x_delta, sign_gradient_x],
                                            feed_dict={x: x_reshape, y_: [fake_label], keep_prob: 1.0})
                x_updated = np.reshape(x_updated, (1, 784))
                result_prob, result = sess.run([y_prob, prediction], feed_dict={x: x_updated, keep_prob: 1.0})
                distort_prob = result_prob[0][result]
            print("finishing image %d ....." % idx)
            visualize_image(pp, image, x_updated[0], delta, chosen_images_probs[idx], distort_prob)

        pp.close()


def reshape(pixels):
    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are float from 0 to 1
    pixels = np.array(pixels, dtype='float64')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))
    return pixels


def visualize_image(pp, original, adversarial, delta, prob_correct, prob_ostrich):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    original_pixels = reshape(original)
    plt.imshow(original_pixels)
    plt.title('original\nwith label {label}\n with probability {prob}'
              .format(label=FLAGS.true_class, prob=round(prob_correct, 3)))
    plt.tight_layout()

    plt.subplot(133)
    adversarial_pixels = reshape(adversarial)
    plt.imshow(adversarial_pixels)
    plt.title('adversarial\nwith label {label}\n with probability {prob}'
              .format(label=FLAGS.target_class, prob=round(prob_ostrich, 3)))
    plt.tight_layout()

    plt.subplot(132)
    delta_pixels = reshape(delta)
    plt.imshow(delta_pixels)
    plt.title('delta')
    plt.tight_layout()

    plt.savefig(pp, format='pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--saved_model', type=str,
                        default='models/non_adversarial/mnist',
                        help='Directory for loading saved model')
    parser.add_argument('--true_class', type=int,
                        default=2,
                        help='images with true class to be adversed')
    parser.add_argument('--target_class', type=int,
                        default=6,
                        help='class chosen images to be adversed to')
    parser.add_argument('--num_images', type=int,
                        default=10,
                        help='number of images to be adversed')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
