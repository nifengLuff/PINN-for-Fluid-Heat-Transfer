"""
Steady State Heat Transfer Model 6: 5*5离散化, CNN, kernel size=3, 输入产热&热导率, 输出T, 由T计算loss
存在数据洗牌
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

#数据集规模
set_num = 10000
batch_size = 1000
input_size = 5
domain_k_q = torch.empty(set_num, 2, input_size, input_size) #k, q

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

            
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
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
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        '''
        self.out = torch.nn.Linear(6*3*3, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        output = x.view(x.size(0), 3, 3)
        return output

criterion = torch.nn.MSELoss()
net = CNN()
opt = torch.optim.Adam(params = net.parameters(), lr = 0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.9)
dataset = TensorDataset(domain_k_q, domain_k_q)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(3000):
    training_loss = 0
    for batch_input, batch_output in data_loader:
        opt.zero_grad()
        output = net(batch_input)
        true_loss = torch.zeros(batch_size)
        l = 0
        with torch.no_grad():
            domain_T = torch.empty(batch_size, input_size, input_size) #T
            for i in range(batch_size):
                for j in range(input_size):
                    for k in range(input_size):
                        if j == 0 or j == input_size - 1 or k == 0 or k == input_size - 1:
                            domain_T[i, j, k] = 0.1
                        else:
                            domain_T[i, j, k] = output[i, j-1, k-1]
        for i in range(1, input_size - 1):
            for j in range(1, input_size - 1):
                l = l + criterion(
                    (domain_T[:, i-1, j] - output[:, i-1, j-1])*(batch_input[:, 0, i-1, j] + batch_input[:, 0, i, j])\
                    + (domain_T[:, i+1, j] - output[:, i-1, j-1])*(batch_input[:, 0, i+1, j] + batch_input[:, 0, i, j])\
                    + (domain_T[:, i, j-1] - output[:, i-1, j-1])*(batch_input[:, 0, i, j-1] + batch_input[:, 0, i, j])\
                    + (domain_T[:, i, j+1] - output[:, i-1, j-1])*(batch_input[:, 0, i, j+1] + batch_input[:, 0, i, j])\
                    + batch_input[:, 1, i, j]
                    , true_loss)
        l.backward()
        opt.step()
        training_loss += l.item()

    print(epoch, training_loss, opt.state_dict()['param_groups'][0]['lr'])
    scheduler.step()

new_input = torch.empty(1, 2, input_size, input_size)
for i in range(2):
    for j in range(input_size):
        for k in range(input_size):
            new_input[0, i, j, k] = 0.5
predicted_output = net(new_input)
print("预测输出:")
print(predicted_output)