import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from utils import *


with open('application.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

print('Config: %s' % config)

# Fixed seed
torch.manual_seed(config['seed'])

if len(sys.argv) < 3:
    print('Parameters missing')
    exit(1)

input_file = sys.argv[1]
print('Input file: %s' % input_file)

model_weights_path = sys.argv[2]
print('Model weights path: %s' % model_weights_path)

df = pd.read_csv(input_file, delimiter=';', header=0, index_col='date', parse_dates=True, dayfirst=True)

# Simplify features
df = df.drop('year', axis=1)
df = df.drop('age', axis=1)
df = df.drop('state', axis=1)
df = df.drop('region', axis=1)
df = df.drop('city', axis=1)
df = df.drop('occupation', axis=1)

# Re-training
training_sample = df.values[0].astype('float32')
x = training_sample[:-1]
y = training_sample[-1:]
consume_max = config['consume']['max'] # np.max(training_set[:, -1:])
consume_min = config['consume']['min'] # np.min(training_set[:, -1:])
y = (y - consume_min) / (consume_max - consume_min)
dataX = Variable(torch.Tensor(np.array(x)))
dataX = dataX.unsqueeze(0).unsqueeze(0)
dataY = Variable(torch.Tensor(np.array(y)))

lstm = torch.load(model_weights_path)

num_epochs = config['hyperparams']['epochs_retraining']
learning_rate = config['hyperparams']['learning_rate']

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm(dataX)
    optimizer.zero_grad()

    loss = criterion(outputs[0], dataY)

    loss.backward()

    optimizer.step()

    if epoch % 9 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

print('Retraining completed')


data = df.values.astype('float32')

# input, _ = sliding_windows(data, 1)

input = data[:, :-1]
input = Variable(torch.Tensor(np.array(input)))
input = input.unsqueeze(1)
real_values = data[:, -1:]
real_values = Variable(torch.Tensor(np.array(real_values)))

lstm.eval()

# Inference
predictions = lstm(input)

predictions = predictions.data.numpy()
real_values = real_values.data.numpy()

predictions = predictions * (1100 - 350) + 350

# print('RMSE: %1.5f' % rmse(dataY_plot, data_predict))

plt.figure(figsize=(12, 6))

plt.plot(real_values, label='Actual Data')
plt.plot(predictions, label='Predicted Data')
plt.title('Consumes Prediction')
plt.legend()
plt.show()

torch.save(lstm, 'checkpoint/model_weights.pth')
