'''
Created on 2020年11月18日

@author: zjf
'''

import torch
import torch.nn as nn

if __name__ == '__main__':
    # Fake NN output
    out = torch.Tensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]])
    
    # Categorical targets
    x = torch.Tensor([[1], [2], [2]])
    
    # Categorical targets
    y = torch.Tensor([1, 2, 0])
    
    
    # Calculating the loss
    loss_val = nn.CrossEntropyLoss()(out, y.long())
    
    print(loss_val)
