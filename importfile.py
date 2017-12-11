import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
