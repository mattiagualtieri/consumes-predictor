import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from lstm import LSTM
import numpy as np
from torch.autograd import Variable


# https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=_BcDEjcABRVz


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def rmse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual, pred)).mean())


df = pd.read_csv('input.csv', delimiter=';', header=0, index_col='date', parse_dates=True, dayfirst=True)

label_encoder = LabelEncoder()
min_max_scaler = MinMaxScaler()

df['state'] = label_encoder.fit_transform(df['state'])
df['region'] = label_encoder.fit_transform(df['region'])
df['city'] = label_encoder.fit_transform(df['city'])
df['occupation'] = label_encoder.fit_transform(df['occupation'])

# Simplify features
df = df.drop('year', axis=1)
df = df.drop('age', axis=1)
df = df.drop('state', axis=1)
df = df.drop('region', axis=1)
df = df.drop('city', axis=1)
df = df.drop('occupation', axis=1)

training_set = df.values
training_set = training_set.astype('float32')

# Scale y in [0, 1]
consume_max = np.max(training_set[:, 4:5])
consume_min = np.min(training_set[:, 4:5])
training_set[:, 4:5] = (training_set[:, 4:5] - consume_min) / (consume_max - consume_min)
training_data = training_set

seq_length = 15
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# Training
num_epochs = 2000
learning_rate = 0.01

input_size = 5
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # Obtain the loss function
    loss = criterion(outputs, trainY[:, 4:5])

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# Evaluation (with train data)
lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()
dataY_plot = dataY_plot[:, 4:5]

# Normalize output
# consume_predicted_max = np.max(data_predict)
# consume_predicted_min = np.min(data_predict)
# data_predict = (data_predict - consume_predicted_min) / (consume_predicted_max - consume_predicted_min)

dataY_plot = dataY_plot * (consume_max - consume_min) + consume_min
data_predict = data_predict * (consume_max - consume_min) + consume_min

print('RMSE: %1.5f' % rmse(dataY_plot, data_predict))

plt.figure(figsize=(15, 6))
plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot, label='Actual Data')
plt.plot(data_predict, label='Predicted Data')
plt.title('Consumes Prediction')
plt.legend()
plt.show()
