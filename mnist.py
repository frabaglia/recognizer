import matplotlib
import tensorflow as tf
import utils
import gzip
import binascii
import struct
import numpy
from config import CONSTANTS
from config import TENSORS

matplotlib.use('agg')

# DESCARGAR O REUTILIZAR MUESTRAS
train_data_filename = utils.maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = utils.maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = utils.maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = utils.maybe_download('t10k-labels-idx1-ubyte.gz')

# EXTRAER IMAGENES
train_data = utils.extract_data(train_data_filename, 60000)
test_data = utils.extract_data(test_data_filename, 10000)

# EXTRAER ETIQUETAS
train_labels = utils.extract_labels(train_labels_filename, 60000)
test_labels = utils.extract_labels(test_labels_filename, 10000)

print('Training labels shape', train_labels.shape)
print('Label vector example', train_labels[0])

# Las imagenes son de 28x28 en blanco y negro definidos en un solo canal [-1:1].
# Por ahora son arreglos de 28 x 28 en formas de listas de 28 elementos x listas de 28 elementos.
validation_data = train_data[:CONSTANTS("NUMBER_OF_VALIDATION_SAMPLES"), :, :, :]
validation_labels = train_labels[:CONSTANTS("NUMBER_OF_VALIDATION_SAMPLES")]

# Las etiquetas son los digitos de 0 a 9 codificados como vectores de 10 elementos.
# Por ahora son arreglos de 28 x 28 en formas de listas de 28 elementos x listas de 28 elementos.
train_data = train_data[CONSTANTS("NUMBER_OF_VALIDATION_SAMPLES"):, :, :, :]
train_labels = train_labels[CONSTANTS("NUMBER_OF_VALIDATION_SAMPLES"):]

train_dataset_size = train_labels.shape[0]
print('Validation shape', validation_data.shape)
print('Train size', train_dataset_size)

# Los placeholders son la forma de ingresar datos a un grafo.

# Por acá vamos a meter las imagenes de entrada y a imponer los labels de salida.
train_data_node = tf.placeholder(   tf.float32,
                                    shape=(CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"), 
                                    CONSTANTS("IMAGE_SIZE"), 
                                    CONSTANTS("IMAGE_SIZE"), 
                                    CONSTANTS("NUM_CHANNELS")))

train_labels_node = tf.placeholder( tf.float32,
                                    shape=(CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"), 
                                    CONSTANTS("NUM_LABELS")))

# Por acá vamos a meter las constantas de prueba y validación en nodos.
# Esto permite aplicarlas dentro del grafo, no podemos hacerlo directamente con un array.
validation_data_node = tf.constant(validation_data)
test_data_node = tf.constant(test_data)

# Training computation: logits + cross-entropy loss.
logits = utils.model(train_data_node, True)
loss =  tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2( labels=train_labels_node, 
                                                    logits=logits))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(TENSORS("fc1_weights")) + tf.nn.l2_loss(TENSORS("fc1_biases")) +
                tf.nn.l2_loss(TENSORS("fc2_weights")) + tf.nn.l2_loss(TENSORS("fc2_biases")))

# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# Controls the learning rate decay.
batch = tf.Variable(0)

# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay( 0.01,                # Base learning rate.
                                            batch * CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"),  # Current index into the dataset.
                                            train_dataset_size,          # Decay step.
                                            0.95,                # Decay rate.
                                            staircase=True)

# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)

# Predictions for the minibatch, validation set and test set.
train_prediction = tf.nn.softmax(logits)

# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = tf.nn.softmax(utils.model(validation_data_node))
test_prediction = tf.nn.softmax(utils.model(test_data_node))
print('Setting up training variables')

# Create a new interactive session that we'll use in
# subsequent code cells.
s = tf.InteractiveSession()

# Use our newly created session as the default for 
# subsequent operations.
s.as_default()

# Initialize all the variables we defined above.
tf.global_variables_initializer().run()


# Grab the first CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH") examples and labels.
batch_data = train_data[:CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"), :, :, :]
batch_labels = train_labels[:CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH")]

# This dictionary maps the batch data (as a numpy array) to the
# node in the graph it should be fed to.
feed_dict = {   
                train_data_node: batch_data,
                train_labels_node: batch_labels
            }

# Run the graph and fetch some of the nodes.
_, l, lr, predictions = s.run(  [optimizer, loss, learning_rate, train_prediction],
                                feed_dict=feed_dict)

print('Trained!')

print(predictions[0])

# The highest probability in the first entry.
print('First prediction', numpy.argmax(predictions[0]))

# But, predictions is actually a list of CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH") probability vectors.
print(predictions.shape)

# So, we'll take the highest probability for each vector.
print('All predictions', numpy.argmax(predictions, 1))

print('Batch labels', numpy.argmax(batch_labels, 1))

correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(batch_labels, 1))
total = predictions.shape[0]

print(float(correct) / float(total))

confusions = numpy.zeros([10, 10], numpy.float32)
bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(batch_labels, 1))
for predicted, actual in bundled:
  confusions[predicted, actual] += 1

# Train over the first 1/4th of our training set.
epochs = train_dataset_size // CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH")

for epoch in range(epochs):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (epoch * CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH")) % (train_dataset_size - CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"))
    batch_data = train_data[offset:(offset + CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH")), :, :, :]
    batch_labels = train_labels[offset:(offset + CONSTANTS("NUMBER_OF_SAMPLES_PER_EPOCH"))]
    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the graph and fetch some of the nodes.
    _, l, lr, predictions = s.run(
      [optimizer, loss, learning_rate, train_prediction],
      feed_dict=feed_dict)
    
    # Print out the loss periodically.
    if epoch % 100 == 0:
        error, _ = utils.error_rate(predictions, batch_labels)
        print('Step %d of %d' % (epoch, epochs))
        print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
        print('Validation error: %.1f%%' % utils.error_rate(
              validation_prediction.eval(), validation_labels)[0])

test_error, confusions = utils.error_rate(test_prediction.eval(), test_labels)
print('Test error: %.1f%%' % test_error)