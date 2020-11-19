# -*- coding:utf-8 -*-
'''
Created on 2020年11月17日

@author: zjf
'''
'''
利用Pytorch实现实现生成莎士比亚风格的文章
参考：https://www.tensorflow.org/tutorials/sequences/text_generation
'''

import torch
import torch.nn as nn
import numpy as np
import sys
import time


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p=probas.ravel())
    return out


def generate_output(indices_char, usr_input, predict):
    generated = usr_input
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    sys.stdout.write(generated)
    
    for _ in range(400):
        x_pred = np.zeros((1, Tx, len(chars)))
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = predict(x_pred)  # 预测下一个字符
        next_index = sample(preds, temperature=1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue

        
if __name__ == '__main__':
    text = open("data/shakespeare.txt").read().lower()
    text = text[37:]  # 去掉标题
    print('文本长度:', len(text))
    
    # 读出文本，形成输入输出结果
    X = []
    Y = []
    Tx = 40
    stride = 3
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])

    print('样本数量:', len(X))
    
    print("x[0]=", X[0])
    print("x[1]=", X[1])
    print("y[0]=", Y[0])

    # 采用one-hot编码
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    m = len(X)
    n = len(chars)
    x = np.zeros((m, Tx, n), dtype=np.float)
    y = np.zeros((m, n), dtype=np.float)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
#         y[i] = char_indices[Y[i]]
     
    # 定义模型   
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
            _, (h_n, _) = self.lstm_layer2(out)
            # h_n大小为num_layers * num_directions, batch, hidden_size，只获取最后一层layer的输出
            out = h_n[-1]
            out = self.dropout_2(out)
            out = self.dense(out)
            out = self.softmax(out)
            return out

    model = Model()
    print(model)

    class MyDataset(torch.utils.data.Dataset):

        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
    
        def __getitem__(self, index):  # 返回的是tensor
            img, target = self.images[index], self.labels[index]
            return img, target
    
        def __len__(self):
            return len(self.images)

    # 训练
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()  # rnn is your model 
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    batch_size = 1280
    train_loader = torch.utils.data.DataLoader(MyDataset(x, y), batch_size=batch_size, shuffle=True)
    batchnum = len(x) / batch_size
    
    for t in range(30):
        for i_batch, batchdata in enumerate(train_loader):
            # Transfer to GPU
            xx, yy = batchdata[0].to(device), batchdata[1].to(device)
            optimizer.zero_grad()
            
            y_pred = model(xx.float())
            loss = loss_function(y_pred, yy.max(1)[1])
            loss.backward()
            optimizer.step()
            print("\r%d/%d" % (i_batch, batchnum), end="", flush=True)
            if i_batch % 10 == 9:
                print(' Loss = ', loss)
                
        print('Train Epoch ', t, ' Loss = ', loss)
    
    # 应用
    generate_output(indices_char, "the sunlight, beautiful", lambda x:model(torch.tensor(x).to(device).float()).cpu().detach().numpy()[0])
    
