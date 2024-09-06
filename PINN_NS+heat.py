import subprocess
import numpy as np
import ast

import torch
#from torch.utils.data import DataLoader, TensorDataset
#调用PINN_NS_pooling2.py
subprocess.run(['python', 'PINN_NS_pooling2.py'])
domain_u = torch.load('u.pth')
domain_v = torch.load('v.pth')

rho = 1
cp = 1
nu = 0.1
#数据集规模
set_num = 10000
#batch_size = 1000
set_num_eval = 1000
input_size = 10
domain_k_q = torch.empty(set_num, 2, input_size, input_size) #k, q
domain_k_q_eval = torch.empty(set_num_eval, 2, input_size, input_size) #k, q

def k_generation():
    k = 0.5 + 0.5*torch.rand(1, 1)
    return k

def q_generation():
    q = torch.rand(1, 1)
    return q

for i in range(set_num):
    for j in range(input_size):
        for k in range(input_size):
            if j == 0 or k == 0 or j == input_size - 1 or k == input_size - 1:
                domain_k_q[i, 0, j, k] = 1.0 #k
                domain_k_q[i, 1, j, k] = 0   #q
            else:
                domain_k_q[i, 0, j, k] = k_generation()
                domain_k_q[i, 1, j, k] = q_generation()
'''
for i in range(set_num_eval):
    for j in range(input_size):
        for k in range(input_size):
            if j == 0 or k == 0 or j == input_size - 1 or k == input_size - 1:
                domain_k_q_eval[i, 0, j, k] = 1.0 #k
                domain_k_q_eval[i, 1, j, k] = 0   #q
            else:
                domain_k_q_eval[i, 0, j, k] = k_generation() #k
                domain_k_q_eval[i, 1, j, k] = q_generation() #q
'''           

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        '''
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        '''
        #self.out1 = torch.nn.Linear(4*5*5, 32)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(4*(input_size - 2)**2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, (input_size - 2)**2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def loss(u, v, T, k, q):
    grad_u_x_m = torch.zeros(input_size, input_size)
    grad_u_x_l = torch.zeros(input_size, input_size)
    grad_u_x_r = torch.zeros(input_size, input_size)

    grad_u_y_m = torch.zeros(input_size, input_size)
    grad_u_y_l = torch.zeros(input_size, input_size)
    grad_u_y_r = torch.zeros(input_size, input_size)

    grad_v_x_m = torch.zeros(input_size, input_size)
    grad_v_x_l = torch.zeros(input_size, input_size)
    grad_v_x_r = torch.zeros(input_size, input_size)

    grad_v_y_m = torch.zeros(input_size, input_size)
    grad_v_y_l = torch.zeros(input_size, input_size)
    grad_v_y_r = torch.zeros(input_size, input_size)

    grad_T_x_m = torch.zeros(input_size, input_size)
    grad_T_x_l = torch.zeros(input_size, input_size)
    grad_T_x_r = torch.zeros(input_size, input_size)

    grad_T_y_m = torch.zeros(input_size, input_size)
    grad_T_y_l = torch.zeros(input_size, input_size)
    grad_T_y_r = torch.zeros(input_size, input_size)
    
    grad_T_x_2 = torch.zeros(input_size, input_size)
    grad_T_y_2 = torch.zeros(input_size, input_size)
    grad_T_x_2 = torch.zeros(input_size, input_size)
    grad_T_y_2 = torch.zeros(input_size, input_size)

    for i in range(1, input_size - 1):
        for j in range(1, input_size - 1):
            grad_u_x_m[i, j] = (u[i, j + 1] - u[i, j - 1]) / 2
            grad_u_y_m[i, j] = (u[i - 1, j] - u[i + 1, j]) / 2
            grad_v_x_m[i, j] = (v[i, j + 1] - v[i, j - 1]) / 2
            grad_v_y_m[i, j] = (v[i - 1, j] - v[i + 1, j]) / 2
            
            grad_u_x_l[i, j] = (u[i, j] - u[i, j - 1])
            grad_u_y_l[i, j] = (u[i, j] - u[i + 1, j])
            grad_v_x_l[i, j] = (v[i, j] - v[i, j - 1])
            grad_v_y_l[i, j] = (v[i, j] - v[i + 1, j])

            grad_u_x_r[i, j] = (u[i, j + 1] - u[i, j])
            grad_u_y_r[i, j] = (u[i - 1, j] - u[i, j])
            grad_v_x_r[i, j] = (v[i, j + 1] - v[i, j])
            grad_v_y_r[i, j] = (v[i - 1, j] - v[i, j])

            grad_T_x_m[i, j] = (T[i, j + 1] - T[i, j - 1]) / 2
            grad_T_y_m[i, j] = (T[i - 1, j] - T[i + 1, j]) / 2
            
            grad_T_x_l[i, j] = (T[i, j] - T[i, j - 1])
            grad_T_y_l[i, j] = (T[i, j] - T[i + 1, j])

            grad_T_x_r[i, j] = (T[i, j + 1] - T[i, j])
            grad_T_y_r[i, j] = (T[i - 1, j] - T[i, j])

            grad_T_x_2[i, j] = T[i, j + 1] + T[i, j - 1] - 2*T[i, j]
            grad_T_y_2[i, j] = T[i + 1, j] + T[i - 1, j] - 2*T[i, j]

    e_m = torch.sum((rho*cp*(u*grad_T_x_m + v*grad_T_y_m) \
         - k*(grad_T_x_2 + grad_T_y_2) - nu*((grad_u_y_m + grad_v_x_m)**2 + 2*(grad_u_x_m**2 + grad_v_y_m**2)) - q)**2)

    e_l = torch.sum((rho*cp*(u*grad_T_x_l + v*grad_T_y_l) \
         - k*(grad_T_x_2 + grad_T_y_2) - nu*((grad_u_y_l + grad_v_x_l)**2 + 2*(grad_u_x_l**2 + grad_v_y_l**2)) - q)**2)

    e_r = torch.sum((rho*cp*(u*grad_T_x_r + v*grad_T_y_r) \
         - k*(grad_T_x_2 + grad_T_y_2) - nu*((grad_u_y_r + grad_v_x_r)**2 + 2*(grad_u_x_r**2 + grad_v_y_r**2)) - q)**2)

    return (e_m + e_l + e_r)

criterion = torch.nn.MSELoss()
net = CNN()
opt = torch.optim.Adam(params = net.parameters(), lr = 0.0001)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.9)
#dataset = TensorDataset(domain_k_q, domain_k_q)
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

l_eval_min = 1000
cnt = 0
for epoch in range(10000):
    #for batch_input, batch_output in data_loader:
    opt.zero_grad()
    output = net(domain_k_q)
    domain_T = torch.zeros(set_num, input_size, input_size)
    #true_loss = torch.zeros(set_num)
    for i in range(set_num):
        for j in range(input_size):
            for k in range(input_size):
                if j == 0 or k == 0 or j == input_size - 1 or k == input_size - 1:
                    domain_T[i, j, k] = 0.1
                else:
                    domain_T[i, j, k] = output[i, (input_size - 2)*(j - 1) + k - 1]
    l = 0
    '''
    with torch.no_grad():
        domain_T = torch.empty(set_num, input_size, input_size) #T
        for i in range(set_num):
            for j in range(input_size):
                for k in range(input_size):
                    if j == 0 or j == input_size - 1 or k == 0 or k == input_size - 1:
                        domain_T[i, j, k] = 0.1
                    else:
                        domain_T[i, j, k] = output[i, (j-1)*3+(k-1)]
    for i in range(1, input_size - 1):
        for j in range(1, input_size - 1):
            l = l + criterion(
                (domain_T[:, i-1, j] - output[:, (i-1)*3+(j-1)])*(domain_k_q[:, 0, i-1, j] + domain_k_q[:, 0, i, j])\
                + (domain_T[:, i+1, j] - output[:, (i-1)*3+(j-1)])*(domain_k_q[:, 0, i+1, j] + domain_k_q[:, 0, i, j])\
                + (domain_T[:, i, j-1] - output[:, (i-1)*3+(j-1)])*(domain_k_q[:, 0, i, j-1] + domain_k_q[:, 0, i, j])\
                + (domain_T[:, i, j+1] - output[:, (i-1)*3+(j-1)])*(domain_k_q[:, 0, i, j+1] + domain_k_q[:, 0, i, j])\
                + domain_k_q[:, 1, i, j]
                , true_loss)
    '''
    for i in range(set_num):
        l = l + loss(domain_u, domain_v, domain_T[i], domain_k_q[i, 0], domain_k_q[i, 1])
    l.backward()
    opt.step()

    #测试集的损失
    '''
    output_eval = net(domain_k_q_eval)
    true_loss_eval = torch.zeros(set_num_eval)
    l_eval = 0
    with torch.no_grad():
        domain_T_eval = torch.empty(set_num_eval, input_size, input_size) #T_eval
        for i in range(set_num_eval):
            for j in range(input_size):
                for k in range(input_size):
                    if j == 0 or j == input_size - 1 or k == 0 or k == input_size - 1:
                        domain_T_eval[i, j, k] = 0.1
                    else:
                        domain_T_eval[i, j, k] = output_eval[i, (j-1)*3+(k-1)]
    for i in range(1, input_size - 1):
        for j in range(1, input_size - 1):
            l_eval = l_eval + criterion(
                (domain_T_eval[:, i-1, j] - output_eval[:, (i-1)*3+(j-1)])*(domain_k_q_eval[:, 0, i-1, j] + domain_k_q_eval[:, 0, i, j])\
                + (domain_T_eval[:, i+1, j] - output_eval[:, (i-1)*3+(j-1)])*(domain_k_q_eval[:, 0, i+1, j] + domain_k_q_eval[:, 0, i, j])\
                + (domain_T_eval[:, i, j-1] - output_eval[:, (i-1)*3+(j-1)])*(domain_k_q_eval[:, 0, i, j-1] + domain_k_q_eval[:, 0, i, j])\
                + (domain_T_eval[:, i, j+1] - output_eval[:, (i-1)*3+(j-1)])*(domain_k_q_eval[:, 0, i, j+1] + domain_k_q_eval[:, 0, i, j])\
                + domain_k_q_eval[:, 1, i, j]
                , true_loss_eval)

    #检测测试集loss是否上升
    if l_eval.item() < l_eval_min:
        l_eval_min = l_eval.item()
        cnt = 0
    else:
        cnt += 1
        if cnt > 20:
            break
    '''
    print('generate temperature:', 'epoch =', epoch, ', loss =', l.item())
    #scheduler.step()

new_input = torch.empty(1, 2, input_size, input_size)
for j in range(input_size):
    for k in range(input_size):
        if j == 0 or k == 0 or j == input_size - 1 or k == input_size - 1:
            new_input[0, 0, j, k] = 1.0
            new_input[0, 1, j, k] = 0
        else:
            new_input[0, 0, j, k] = 0.8
            new_input[0, 1, j, k] = 0.8

predicted_output = net(new_input)
print("预测输出:")
print(predicted_output)