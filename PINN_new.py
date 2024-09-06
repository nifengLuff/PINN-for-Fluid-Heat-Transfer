"""
Steady State Heat Transfer Model 6: 4*4离散化, 输入产热&热导率, 输出T, 由T计算loss
"""
import numpy as np
import torch
import random

#数据集规模
set_num = 10000
domain = torch.empty(set_num, 4, 4, 3).double() #k, q, T

def k_generation():
    k = 0.5 + 0.5*torch.rand(1, 1)
    return k

def q_generation():
    q = torch.rand(1, 1)
    return q

for i in range(set_num):
    for j in range(4):
        for k in range(4):
            if j == 0 or k == 0 or j == 3 or k == 3:
                domain[i, j, k, 2] = 0.1
                domain[i, j, k, 1] = 0
                domain[i, j, k, 0] = 1.0
            else:
                domain[i, j, k, 0] = k_generation()
                domain[i, j, k, 1] = q_generation()
            
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.net(x)

loss = torch.nn.MSELoss()
u = MLP()
opt = torch.optim.Adam(params = u.parameters(), lr = 0.0001)

true_T = torch.zeros(set_num, 4)
for epoch in range(3000):
    opt.zero_grad()
    input = domain[:, 1:3, 1:3, 0:2].reshape(set_num,-1).float()
    output = u(input)
    with torch.no_grad():
        for i in range(set_num):       
            domain[i, 1, 1, 2] = output[i, 0]
            domain[i, 1, 2, 2] = output[i, 1]
            domain[i, 2, 1, 2] = output[i, 2]
            domain[i, 2, 2, 2] = output[i, 3]
        for i in range(set_num):
            for j in range(1,3):
                for k in range(1,3):
                    true_T[i, 2*j+k-3] = (domain[i, j, k, 1] +\
                                    (domain[i, j, k, 0] + domain[i, j-1, k, 0])*domain[i, j-1, k, 2] +\
                                    (domain[i, j, k, 0] + domain[i, j+1, k, 0])*domain[i, j+1, k, 2] +\
                                    (domain[i, j, k, 0] + domain[i, j, k-1, 0])*domain[i, j, k-1, 2] +\
                                    (domain[i, j, k, 0] + domain[i, j, k+1, 0])*domain[i, j, k+1, 2]\
                                    )/(4*domain[i, j, k, 0] + domain[i, j-1, k, 0] + domain[i, j+1, k, 0] + domain[i, j, k-1, 0] + domain[i, j, k+1, 0])    
    l = loss(true_T, output)
    print(epoch, l.item())
    l.backward()
    opt.step()

new_input = 0.5*torch.ones(8)
predicted_output = u(new_input)
print("预测输出:")
print(predicted_output)