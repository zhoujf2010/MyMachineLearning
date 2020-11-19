# -*- coding:utf-8 -*-
"""
Description:

@author: WangLeAi
@date: 2019/5/6
"""
# import tensorflow as tf
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(40, 38)),
#     tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(128),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(38),
#     tf.keras.layers.Activation('softmax')
# ])
#
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.summary()

import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm_layer1 = nn.LSTM(input_size=38, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.lstm_layer2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dense = nn.Linear(128, 38)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        output, (_, _) = self.lstm_layer1(x)
        out = self.dropout_1(output)
        _, (h_n, c_n) = self.lstm_layer2(out)
        # h_n大小为num_layers * num_directions, batch, hidden_size，只获取最后一层layer的输出
        out = h_n[-1]
        print(out.shape)
        out = self.dropout_2(out)
        out = self.dense(out)
        out = self.softmax(out)
        return out


temp_model = Model()
# summary(temp_model, (40, 38))
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(temp_model.parameters())

# 训练中才可以使用loss和optimizer，细节参数没调整，大致结构应该相同。
# X_train_tensor = torch.tensor(np.zeros([1, 40, 38]))
X_train_tensor = torch.tensor(np.zeros([1, 40, 38]).tolist())
# X_train_tensor2 = torch.randn(1, 40, 38)

Y_train_tensor = torch.tensor([5])
for t in range(10):
    y_pred = temp_model(X_train_tensor)
    print("predict class:", torch.max(y_pred, 1)[1][0])
    loss = loss_function(y_pred, Y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Train Epoch ', t, ' Loss = ', loss)
