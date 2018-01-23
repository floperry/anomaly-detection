from importfile import *
from keras.models import Model, Input
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras.callbacks import TensorBoard
from time import time
from sklearn import manifold
from sklearn.svm import OneClassSVM

# Load data
data = sio.loadmat("data/FM/train_data_psd.mat")
data_train = np.asarray(data['train_data'])
data = sio.loadmat("data/FM/awgn/testdata_psd_awgn_27dB.mat")
data_test = np.asarray(data['test_data'])

# Data preprocessing
scalar = MinMaxScaler(feature_range=(0, 1))
x_train = scalar.fit_transform(data_train[0:50000, :])
x_valid = scalar.transform(data_train[90000:100000, :])
x_test = scalar.transform(data_test)

# Network parameters
num_input = 512
num_hidden_1 = 8
epochs = 200
optimizer = optimizers.Adadelta()
model_autoencoder_path = "model_svm/autoencoder_1Layer_9.h5"
model_encoder_path = "model_svm/encoder_1Layer_9.h5"

# Define autoencoder
inputs = Input(shape=(num_input,))
encoded_1 = Dense(num_hidden_1, activation='relu')(inputs)
decoded_1 = Dense(num_input, activation='linear')(encoded_1)
autoencoder = Model(inputs=inputs, outputs=decoded_1)
encoder = Model(inputs=inputs, outputs=encoded_1)
autoencoder.compile(optimizer=optimizer, loss='mse')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Network training
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=512,
                shuffle=True,
                verbose=2,
                callbacks=[tensorboard],
                validation_data=(x_valid, x_valid))

# Save model
autoencoder.save(model_autoencoder_path)
encoder.save(model_encoder_path)

# Testing
encoded_train = encoder.predict(x_train)
encoded_test = encoder.predict(x_test)

# Fit oneclasssvm
clf = OneClassSVM(nu=0.1, kernel='linear', gamma=0.1)
clf.fit(encoded_train)
score = clf.decision_function(encoded_test)

# Compute AUC
y_test = np.concatenate((np.zeros(2000), np.ones(2000)))
fpr, tpr, threshold = roc_curve(y_test, score, pos_label=1)
auc_value = auc(fpr, tpr)
print(auc_value)

'''
# Scale and visualize the embedding vectors
def plot_embedding(x, y, title=None):
    # x_min, x_max = np.min(x, 0), np.max(x, 0)
    # x = (x - x_min) / (x_max - x_min)
    scalar_1 = MinMaxScaler(feature_range=(0, 1))
    x = scalar_1.fit_transform(x)

    plt.figure()
    # colors = itertools.cycle(["r", "b"])
    colors = ["r", "b"]
    for i in range(x.shape[0]):
        plt.text(x[i, 0], x[i, 1], str(int(y[i])),
                 color=colors[int(y[i])],
                 fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


X = encoded_test
Y = np.concatenate((np.zeros(2000), np.ones(2000)))

# t-SNE embedding of dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne, Y,
               "t-SNE embedding (time %.2fs)" % (time() - t0))
plt.show()
'''

