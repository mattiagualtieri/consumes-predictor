import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.autograd import Variable

from model.lstm import LSTM
from utils import *


with open('application.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

print('Config: %s' % config)

torch.manual_seed(config['seed'])

if len(sys.argv) < 2:
    print('Input file parameter missing')
    exit(1)

input_file = sys.argv[1]
print('Input file: %s' % input_file)

df = pd.read_csv(input_file, delimiter=';', header=0, index_col='date', parse_dates=True, dayfirst=True)

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
consume_max = np.max(training_set[:, -1:])
consume_min = np.min(training_set[:, -1:])
training_set[:, -1:] = (training_set[:, -1:] - consume_min) / (consume_max - consume_min)
training_data = training_set

seq_length = config['hyperparams']['sequence_length']
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * (1 - config['test_size']))
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# Training
num_epochs = config['hyperparams']['epochs']
learning_rate = config['hyperparams']['learning_rate']

input_size = training_set.shape[1]
hidden_size = config['hyperparams']['lstm_depth']
num_layers = 1
num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    loss = criterion(outputs, trainY[:, -1:])

    loss.backward()

    optimizer.step()
    if epoch % 500 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# Evaluation
lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()
dataY_plot = dataY_plot[:, -1:]

dataY_plot = dataY_plot * (consume_max - consume_min) + consume_min
data_predict = data_predict * (consume_max - consume_min) + consume_min

print('RMSE: %1.5f' % rmse(dataY_plot, data_predict))

plt.figure(figsize=(12, 6))
plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot, label='Actual Data')
plt.plot(data_predict, label='Predicted Data')
plt.title('Consumes Prediction')
plt.legend()
plt.show()
