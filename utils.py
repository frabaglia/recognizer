import numpy, os, gzip, tensorflow as tf
from six.moves.urllib.request import urlretrieve
from config import CONSTANTS
from config import TENSORS

# TF Aliases
conv2d = tf.nn.conv2d
relu = tf.nn.relu
bias_add = tf.nn.bias_add
max_pool = tf.nn.max_pool
dropout = tf.nn.dropout

"""
Return the error rate and confusions.
"""
def error_rate(predictions, labels):
    correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.float32)
    bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions

"""
A helper to download the data files if not present.
"""
def maybe_download(filename):
    if not os.path.exists(CONSTANTS("WORK_DIRECTORY")):
        os.mkdir(CONSTANTS("WORK_DIRECTORY"))
    filepath = os.path.join(CONSTANTS("WORK_DIRECTORY"), filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(CONSTANTS("SOURCE_URL") + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

"""
Extract the images into a 4D tensor [image index, y, x, channels].
  
For MNIST data, the number of channels is always 1.

Values are rescaled from [0, 255] down to [-0.5, 0.5].
"""
def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(CONSTANTS("IMAGE_SIZE") * CONSTANTS("IMAGE_SIZE") * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (CONSTANTS("PIXEL_DEPTH") / 2.0)) / CONSTANTS("PIXEL_DEPTH")
        data = data.reshape(num_images, CONSTANTS("IMAGE_SIZE"), CONSTANTS("IMAGE_SIZE"), 1)
        return data

"""
Extract the labels into a 1-hot matrix [image index, label index].
"""
def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(CONSTANTS("NUM_LABELS")) == labels[:, None]).astype(numpy.float32)

"""
The Model definition.

2D convolution, with 'SAME' padding (i.e. the output feature map has
the same size as the input). Note that {strides} is a 4D array whose
shape matches the data layout: [image index, y, x, depth].
"""
def model(data, train=False):

    """
    2D Convolution: tf.nn.conv2d

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

    Args:
        input: A Tensor. Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
        filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

    Return:
    """
    conv = conv2d(data,
                        TENSORS("conv1_weights"),
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # Bias and rectified linear non-linearity.
    relu_bias = relu(bias_add(conv, TENSORS("conv1_biases")))

    # Max pooling. The kernel size spec ksize also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = max_pool(relu_bias,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    conv = conv2d(pool,
                        TENSORS("conv2_weights"),
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    relu_bias = relu(bias_add(conv, TENSORS("conv2_biases")))
    pool = max_pool(relu_bias,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = relu(tf.matmul(reshape, TENSORS("fc1_weights")) + TENSORS("fc1_biases"))

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = dropout(hidden, 0.5, seed=CONSTANTS("SEED"))
    return tf.matmul(hidden, TENSORS("fc2_weights")) + TENSORS("fc2_biases")