from importfile import *

# Load data
data = sio.loadmat("data/FM/traindata_psd_10000.mat")
data_train = np.asarray(data['train_data']).transpose()
data = sio.loadmat("data/FM/interference/testdata_psd_interference_m10dB.mat")
data_test = np.asarray(data['test_data']).transpose()


# Data preprocessing
scalar = MinMaxScaler()
x_train = scalar.fit_transform(data_train)
x_test = scalar.transform(data_test)


# Training Parameters
learning_rate = 0.2
num_steps = 500
batch_size = 256

display_step = 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 16    # 1st layer num features
num_hidden_2 = 2    # 2nd layer features (the latent dim)
num_input = 512    # data input

# tf Graph input
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
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Define KL divergence
def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) \
           - (1 - p) * tf.log(1 - p_hat)


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Start Training
# Start a new session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Store loss
    l_store, n_store = [], []
    # Training
    for i in range(1, num_steps+1):
        # Run optimization op and cost op
        _, l = sess.run([optimizer, loss], feed_dict={X: x_train})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step % i: Minibatch Loss: %f' % (i, l))
            l_store.append(l)
            n_store.append(i)

    # Save model
    saver = tf.train.Saver()
    model_path = "model/model.ckpt"
    save_path = saver.save(sess, model_path)

    # Testing
    # Encode and decode data from test set
    canvas_orig = np.empty((2000, 512))
    canvas_recon = np.empty((2000, 512))
    canvas_encode = np.empty((2000, 2))
    score = []
    # Encode and decode data
    for i in range(2000):
        batch_x = np.asarray(x_test[i]).reshape([1, 512])
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        # Encode data
        f = sess.run(encoder_op, feed_dict={X: batch_x})

        # Compute MSE
        score.append(mean_squared_error(batch_x, g))

        canvas_orig[i, :] = scalar.inverse_transform(batch_x)
        canvas_recon[i, :] = scalar.inverse_transform(g)
        canvas_encode[i, :] = f

    y_test = np.concatenate((np.zeros(1000), np.ones(1000)))
    fpr, tpr, threshold = roc_curve(y_test, score, pos_label=1)
    auc_value = auc(fpr, tpr)
    print(auc_value)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.show()

    '''
    # Save figure to pdf
    pp = PdfPages("result/bar.pdf")
    # Plot normal data
    plot1 = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(canvas_orig[1, :])
    plt.title("Normal: original data")
    plt.subplot(2, 1, 2)
    plt.plot(canvas_recon[1, :])
    plt.title("Normal: reconstructed data")
    pp.savefig(plot1)
    plt.close()
    # Plot Abnormal data
    plot2 = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(canvas_orig[1001, :])
    plt.title("Abnormal: original data")
    plt.subplot(2, 1, 2)
    plt.plot(canvas_recon[1001, :])
    plt.title("Abnormal: reconstructed data")
    pp.savefig(plot2)
    plt.close()
    # Plot loss
    plot3 = plt.figure()
    plt.plot(n_store, l_store)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    pp.savefig(plot3)
    plt.close()
    pp.close()

    # Plot time-frequency figure
    plt.figure()
    plt.subplot(2, 1, 1)
    sns.heatmap(data_train, cmap="YlGnBu")
    plt.subplot(2, 1, 2)
    sns.heatmap(data_test, cmap="YlGnBu")

    plt.figure()
    plt.subplot(3, 1, 1)
    sns.heatmap(canvas_orig, cmap="YlGnBu")
    plt.subplot(3, 1, 2)
    sns.heatmap(canvas_recon, cmap="YlGnBu")
    plt.subplot(3, 1, 3)
    sns.heatmap(canvas_encode, cmap="YlGnBu")

    plt.show()
    '''






