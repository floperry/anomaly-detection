from importfile import *

# Load data
data = sio.loadmat("data/FM/train_data_psd.mat")
data_train = np.asarray(data['train_data'])
data = sio.loadmat("data/FM/awgn/testdata_psd_awgn_24dB.mat")
data_test = np.asarray(data['test_data'])

# Data preprocessing
scalar = MinMaxScaler(feature_range=(0, 1))
x_train = scalar.fit_transform(data_train[0:50000, :])
x_valid = scalar.transform(data_train[90000:100000, :])
x_test = scalar.transform(data_test)

# Training Parameters
learning_rate = 1
num_steps = 150
batch_size = 256
beta = 0.1
display_step = 2
examples_to_show = 10

# Network Parameters
num_hidden_1 = 64    # 1st layer num features
num_hidden_2 = 32    # 2nd layer num features
num_hidden_3 = 2   # 3rd layer num features
num_hidden_4 = 32    # 4th layer num features
num_input = 512    # data input

# tf Graph input
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    return layer_3


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
regularizer = tf.nn.l2_loss(weights['encoder_h1']) + tf.nn.l2_loss(weights['encoder_h2']) \
              + tf.nn.l2_loss(weights['encoder_h3']) + tf.nn.l2_loss(weights['decoder_h1']) \
              + tf.nn.l2_loss(weights['decoder_h2']) + tf.nn.l2_loss(weights['decoder_h3']) \
              + tf.nn.l2_loss(biases['encoder_b1']) + tf.nn.l2_loss(biases['encoder_b2']) \
              + tf.nn.l2_loss(biases['encoder_b3']) + tf.nn.l2_loss(biases['decoder_b1']) \
              + tf.nn.l2_loss(biases['decoder_b2']) + tf.nn.l2_loss(biases['decoder_b3'])
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2) + beta * regularizer)
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
    loss_valid = []
    # Training
    for i in range(1, num_steps+1):
        # Shuffle data
        np.random.shuffle(x_train)
        # Run optimization op and cost op
        _, l = sess.run([optimizer, loss], feed_dict={X: x_train})

        # Validation
        x_recon = sess.run(decoder_op, feed_dict={X: x_valid})
        l_valid = mean_squared_error(x_valid, x_recon)
        # Display logs per step
        if i % display_step == 0:
            print('Step % i: Minibatch Loss: %f, Valid Loss: %f' % (i, l, l_valid))
            l_store.append(l)
            n_store.append(i)
            loss_valid.append(l_valid)

    # Save model
    saver = tf.train.Saver()
    model_path = "model/model_3Layer_7.ckpt"
    save_path = saver.save(sess, model_path)

    # Testing
    # Encode and decode data from test set
    canvas_orig = np.empty((4000, 512))
    canvas_recon = np.empty((4000, 512))
    # canvas_encode = np.empty((4000, 2))
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
        # canvas_encode[i, :] = f

    y_test = np.concatenate((np.zeros(2000), np.ones(2000)))
    fpr, tpr, threshold = roc_curve(y_test, score, pos_label=1)
    auc_value = auc(fpr, tpr)
    print(auc_value)
    '''
    plt.figure()
    plt.plot(fpr, tpr)
    plt.show()   
    '''

    '''
    # Save figure to pdf
    pp = PdfPages("result/bar3.pdf")
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
    plt.plot(canvas_orig[2001, :])
    plt.title("Abnormal: original data")
    plt.subplot(2, 1, 2)
    plt.plot(canvas_recon[2001, :])
    plt.title("Abnormal: reconstructed data")
    pp.savefig(plot2)
    plt.close()
    # Plot loss
    plot3 = plt.figure()
    plt.plot(n_store, l_store, 'r')
    plt.plot(n_store, loss_valid, 'b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    pp.savefig(plot3)
    plt.close()
    pp.close()
    '''
    '''
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






