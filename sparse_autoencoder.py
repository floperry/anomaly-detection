from importfile import *

# Load data
data = sio.loadmat("data/FM/train_data_psd.mat")
data_train = np.asarray(data['train_data'])
data = sio.loadmat("data/FM/awgn/testdata_psd_awgn_24dB.mat")
data_test = np.asarray(data['test_data'])


# Data preprocessing
scalar = MinMaxScaler()
x_train = scalar.fit_transform(data_train[0:10000, :])
x_valid = scalar.transform(data_train[90000:100000, :])
x_test = scalar.transform(data_test)

# Parameters of network
num_input = 512
num_hidden_1 = 128
num_hidden_2 = 2
learning_rate = 0.1
epochs = 20
batch_size = 256


# Define Layers
input_layer = Input(shape=(num_input,))
hidden_1 = Dense(num_hidden_1, activation='relu',
                 activity_regularizer=regularizers.l1(0))(input_layer)
hidden_2 = Dense(num_hidden_2, activation='relu',
                 activity_regularizer=regularizers.l1(0))(hidden_1)
hidden_3 = Dense(num_hidden_1, activation='relu',
                 activity_regularizer=regularizers.l1(0))(hidden_2)
output_layer = Dense(num_input, activation='relu')(hidden_3)

# Define Optimizer and Loss
optimizer = optimizers.RMSprop(lr=learning_rate)
loss = losses.mean_squared_error

# Model Setup
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=optimizer, loss=loss)
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=2,
                validation_data=(x_valid, x_valid))

score = []
for i in range(len(x_test)):
    # Predict
    test_x = np.asarray(x_test[i]).reshape([1, 512])
    x_test_recon = autoencoder.predict(test_x)

    # Compute MSE
    score.append(mean_squared_error(test_x, x_test_recon))

# Compute AUC
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






