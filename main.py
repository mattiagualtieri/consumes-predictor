import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from lstm import LSTM

# https://cnvrg.io/pytorch-lstm/
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
# https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
# https://www.educative.io/answers/how-to-build-an-lstm-model-using-pytorch
# https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=_BcDEjcABRVz

df = pd.read_csv('input.csv', delimiter=';', header=0, index_col='date', parse_dates=True)
# plt.style.use('ggplot')
# df['consume'].plot(label='Consume', title='Consumes')
X = df.iloc[:, :-1]
y = df.iloc[:, df.shape[1] - 1:df.shape[1]]

label_encoder = LabelEncoder()
min_max_scaler = MinMaxScaler()

X['state'] = label_encoder.fit_transform(X['state'])
X['region'] = label_encoder.fit_transform(X['region'])
X['city'] = label_encoder.fit_transform(X['city'])
X['occupation'] = label_encoder.fit_transform(X['occupation'])

# Simplify features
X = X.drop('year', axis=1)
X = X.drop('age', axis=1)
X = X.drop('state', axis=1)
X = X.drop('region', axis=1)
X = X.drop('city', axis=1)
X = X.drop('occupation', axis=1)

# Scale y in [0, 1]
y = min_max_scaler.fit_transform(y)

X_train, X_test = train_test_split(X, test_size=0.25)
y_train, y_test = train_test_split(y, test_size=0.25)

X_tensors = torch.Tensor(X.values)
X_train_tensors = torch.Tensor(X_train.values)
X_test_tensors = torch.Tensor(X_test.values)
y_tensors = torch.Tensor(y)
y_train_tensors = torch.Tensor(y_train)
y_test_tensors = torch.Tensor(y_test)

X_tensors_final = torch.reshape(X_tensors, (X_tensors.shape[0], 1, X_tensors.shape[1]))
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
y_train_tensors_final = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1, y_train_tensors.shape[1]))
y_test_tensors_final = torch.reshape(y_test_tensors, (y_test_tensors.shape[0], 1, y_test_tensors.shape[1]))


num_epochs = 2000
learning_rate = 0.01
input_dim = X.shape[1]
hidden_dim = 100
output_dim = 1
layer_dim = 1

lstm = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
print(lstm)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    outputs = lstm.forward(X_train_tensors_final)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, backprop
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

lstm.eval()

# forward pass
train_predict = lstm(X_tensors_final)
data_predict = train_predict.data.numpy() #numpy conversion
dataY_plot = y_tensors

data_predict = min_max_scaler.inverse_transform(data_predict) #reverse transformation
dataY_plot = min_max_scaler.inverse_transform(dataY_plot)
plt.figure(figsize=(15, 6))
plt.axvline(x=X.shape[0], c='r', linestyle='--')

plt.plot(dataY_plot, label='Actual Data')
plt.plot(data_predict, label='Predicted Data')
plt.title('Consumes Prediction')
plt.legend()
plt.show()

print('Terminated')
