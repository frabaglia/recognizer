import tensorflow as tf
# The variables below hold all the trainable weights. For each, the
# parameter defines how the variables will be initialized.
IMAGE_SIZE = 28
NUM_LABELS = 10
SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
WORK_DIRECTORY = "/tmp/mnist-data"
PIXEL_DEPTH = 255
NUM_CHANNELS = 1
SEED = 42
NUMBER_OF_SAMPLES_PER_EPOCH = 60
NUMBER_OF_VALIDATION_SAMPLES = 5000

conv1_weights = tf.Variable(
                    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                    stddev=0.1,
                    seed=SEED))

conv1_biases = tf.Variable(tf.zeros([32]))

conv2_weights = tf.Variable(
                    tf.truncated_normal([5, 5, 32, 64],
                    stddev=0.1,
                    seed=SEED))

conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(  # fully connected, depth 512.
                    tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                    stddev=0.1,
                    seed=SEED))

fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

fc2_weights = tf.Variable(
                    tf.truncated_normal([512, NUM_LABELS],
                    stddev=0.1,
                    seed=SEED))

fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

constants = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "NUM_LABELS": NUM_LABELS,
    "SOURCE_URL": SOURCE_URL,
    "WORK_DIRECTORY": WORK_DIRECTORY,
    "PIXEL_DEPTH": PIXEL_DEPTH,
    "NUM_CHANNELS": NUM_CHANNELS,
    "SEED": SEED,
    "NUMBER_OF_SAMPLES_PER_EPOCH": NUMBER_OF_SAMPLES_PER_EPOCH,
    "NUMBER_OF_VALIDATION_SAMPLES": NUMBER_OF_VALIDATION_SAMPLES
}

tensors = {
    "conv1_weights": conv1_weights,
    "conv1_biases": conv1_biases,
    "conv2_weights": conv2_weights,
    "conv2_biases": conv2_biases,
    "fc1_weights": fc1_weights,
    "fc1_biases": fc1_biases,
    "fc2_weights": fc2_weights,
    "fc2_biases": fc2_biases
}

def CONSTANTS(key):
    return constants.get(key)

def TENSORS(key):
    return tensors.get(key)