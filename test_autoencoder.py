from importfile import *

# Load data
data = sio.loadmat("data/FM/train_data_psd.mat")
data_train = np.asarray(data['train_data'])
data = sio.loadmat("data/FM/dsqpsk/testdata_psd_dsqpsk_24dB.mat")
data_test = np.asarray(data['test_data'])

# Data preprocessing
scalar = MinMaxScaler()
x_train = scalar.fit_transform(data_train[0:50000, :])
x_test = scalar.transform(data_test)

# Network Parameters
num_hidden_1 = 128    # 1st layer num features
num_hidden_2 = 2   # 2nd layer num features
num_input = 512    # data input

# Define variables
X = tf.placeholder("float", [None, num_input])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data
y_true = X

# Load model
sess = tf.Session()
saver = tf.train.Saver()
model_path = "model/denoising_model_2Layer_0.ckpt"
saver.restore(sess, model_path)

# Testing
# Encode and decode data from test set
canvas_orig = np.empty((50000, 512))
canvas_recon = np.empty((50000, 512))
canvas_encode = np.empty((4000, 2))
score = []
# Encode and decode data
for i in range(4000):
    batch_x = np.asarray(x_test[i]).reshape([1, 512])
    g = sess.run(decoder_op, feed_dict={X: batch_x})
    # Encode data
    f = sess.run(encoder_op, feed_dict={X: batch_x})

    # Compute MSE
    score.append(mean_squared_error(batch_x, g))

    canvas_orig[i, :] = scalar.inverse_transform(batch_x)
    canvas_recon[i, :] = scalar.inverse_transform(g)
    canvas_encode[i, :] = f

y_test = np.concatenate((np.zeros(2000), np.ones(2000)))
fpr, tpr, threshold = roc_curve(y_test, score, pos_label=1)
auc_value = auc(fpr, tpr)
print(auc_value)

# sio.savemat("fmout_AE1.mat", {'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
'''
plt.figure()
plt.plot(fpr, tpr)
plt.title(u"ROC曲线")
plt.show()
'''



