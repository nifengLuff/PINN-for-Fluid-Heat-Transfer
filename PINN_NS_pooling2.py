"""
NS equation: CNN, 输入drag, 输出u,v,p, 由u,v,p计算loss
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
#数据集规模
rho= 1
nu = 0.1
set_num = 100
set_num_eval = 1
input_size = 10
epoch_num = 10000
domain_drag = torch.zeros(set_num, 1, input_size, input_size)
domain_drag_eval = torch.zeros(set_num_eval, 1, input_size, input_size)

def drag_generation():
    drag = 0.05*torch.rand(1, 1) + 0.05
    return drag

for i in range(set_num):
    for j in range(input_size):
        for k in range(input_size):
            if not(j == 0) and not(k == 0) and not(j == input_size - 1) and not(k == input_size - 1):
                domain_drag[i, 0, j, k] = drag_generation()

for i in range(set_num_eval):
    for j in range(input_size):
        for k in range(input_size):
            if not(j == 0) and not(k == 0) and not(j == input_size - 1) and not(k == input_size - 1):
                domain_drag_eval[i, 0, j, k] = 0.1
            
class pooling_net(torch.nn.Module):
    def __init__(self):
        super(pooling_net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
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
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(4*(input_size - 2)*(input_size - 2), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2*(input_size - 1)**2 + (input_size - 2)**2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def loss(u, v, p, drag):
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

    grad_p_x_m = torch.zeros(input_size, input_size)
    grad_p_x_l = torch.zeros(input_size, input_size)
    grad_p_x_r = torch.zeros(input_size, input_size)

    grad_p_y_m = torch.zeros(input_size, input_size)
    grad_p_y_l = torch.zeros(input_size, input_size)
    grad_p_y_r = torch.zeros(input_size, input_size)
    
    grad_u_x_2 = torch.zeros(input_size, input_size)
    grad_u_y_2 = torch.zeros(input_size, input_size)
    grad_v_x_2 = torch.zeros(input_size, input_size)
    grad_v_y_2 = torch.zeros(input_size, input_size)

    for i in range(1, input_size - 1):
        for j in range(1, input_size - 1):
            grad_u_x_m[i, j] = (u[i, j + 1] - u[i, j - 1]) / 2
            grad_u_y_m[i, j] = (u[i - 1, j] - u[i + 1, j]) / 2
            grad_v_x_m[i, j] = (v[i, j + 1] - v[i, j - 1]) / 2
            grad_v_y_m[i, j] = (v[i - 1, j] - v[i + 1, j]) / 2
            grad_p_x_m[i, j] = (p[i, j + 1] - p[i, j - 1]) / 2
            grad_p_y_m[i, j] = (p[i - 1, j] - p[i + 1, j]) / 2
            
            grad_u_x_l[i, j] = (u[i, j] - u[i, j - 1])
            grad_u_y_l[i, j] = (u[i, j] - u[i + 1, j])
            grad_v_x_l[i, j] = (v[i, j] - v[i, j - 1])
            grad_v_y_l[i, j] = (v[i, j] - v[i + 1, j])
            grad_p_x_l[i, j] = (p[i, j] - p[i, j - 1])
            grad_p_y_l[i, j] = (p[i, j] - p[i + 1, j])

            grad_u_x_r[i, j] = (u[i, j + 1] - u[i, j])
            grad_u_y_r[i, j] = (u[i - 1, j] - u[i, j])
            grad_v_x_r[i, j] = (v[i, j + 1] - v[i, j])
            grad_v_y_r[i, j] = (v[i - 1, j] - v[i, j])
            grad_p_x_r[i, j] = (p[i, j + 1] - p[i, j])
            grad_p_y_r[i, j] = (p[i - 1, j] - p[i, j])

            grad_u_x_2[i, j] = u[i, j + 1] + u[i, j - 1] - 2*u[i, j]
            grad_u_y_2[i, j] = u[i + 1, j] + u[i - 1, j] - 2*u[i, j]
            grad_v_x_2[i, j] = v[i, j + 1] + v[i, j - 1] - 2*v[i, j]
            grad_v_y_2[i, j] = v[i - 1, j] + v[i + 1, j] - 2*v[i, j]

             

    e1_m = torch.sum((u*grad_u_x_m + v*grad_u_y_m + 1/rho*(grad_p_x_m) - nu*(grad_u_x_2 + grad_u_y_2) + drag*u)**2)
    e2_m = torch.sum((u*grad_v_x_m + v*grad_v_y_m + 1/rho*(grad_p_y_m) - nu*(grad_v_x_2 + grad_v_y_2) + drag*v)**2)
    e3_m = torch.sum((grad_u_x_m + grad_v_y_m)**2)

    e1_l = torch.sum((u*grad_u_x_l + v*grad_u_y_l + 1/rho*(grad_p_x_l) - nu*(grad_u_x_2 + grad_u_y_2) + drag*u)**2)
    e2_l = torch.sum((u*grad_v_x_l + v*grad_v_y_l + 1/rho*(grad_p_y_l) - nu*(grad_v_x_2 + grad_v_y_2) + drag*v)**2)
    e3_l = torch.sum((grad_u_x_l + grad_v_y_l)**2)

    e1_r = torch.sum((u*grad_u_x_r + v*grad_u_y_r + 1/rho*(grad_p_x_r) - nu*(grad_u_x_2 + grad_u_y_2) + drag*u)**2)
    e2_r = torch.sum((u*grad_v_x_r + v*grad_v_y_r + 1/rho*(grad_p_y_r) - nu*(grad_v_x_2 + grad_v_y_2) + drag*v)**2)
    e3_r = torch.sum((grad_u_x_r + grad_v_y_r)**2)

    return (e1_m + e2_m + e3_m + e1_l + e2_l + e3_l + e1_r + e2_r + e3_r)

criterion = torch.nn.MSELoss()
net = pooling_net()
opt = torch.optim.Adam(params = net.parameters(), lr = 0.0001)

l_min = 10000
cnt = 0
loss_values = []
for epoch in range(epoch_num):
    opt.zero_grad()
    output = net(domain_drag)
    l = 0
    domain_u = torch.empty(set_num, input_size, input_size)
    domain_v = torch.empty(set_num, input_size, input_size)
    domain_p = torch.empty(set_num, input_size, input_size)
    for i in range(set_num):
        for j in range(input_size):
            for k in range(input_size):
                if k == input_size - 1:
                    domain_p[i, j, k] = 1
                else:
                    domain_p[i, j, k] = output[i, j*(input_size - 1) + k]
                
                if j == 0 or j == input_size - 1 or k == 0 or k == input_size - 1:
                    domain_v[i, j, k] = 0
                else:
                    domain_v[i, j, k] = output[i, 2*(input_size - 1)*(input_size - 1) + (input_size - 2)*(j - 1) + k - 1]
                
                if not(j == 0) and not(j == input_size - 1) and not(k == 0):
                    domain_u[i, j, k] = output[i, (input_size)*(input_size - 1) + (input_size - 1)*(j - 1) + k - 1]
                elif k == 0:
                    if j < 2:
                        domain_u[i, j, k] = 1
                    else:
                        domain_u[i, j, k] = 0

    for i in range(set_num):
        for j in range(input_size):
            for k in range(input_size):
                if j == 0:
                    domain_u[i, j, k] = domain_u[i, j + 1, k]
                elif j == input_size - 1:
                    domain_u[i, j, k] = domain_u[i, j - 1, k]

    for i in range(set_num):
        l = l + loss(domain_u[i], domain_v[i], domain_p[i], domain_drag[i])
    
    l.backward()
    opt.step()
    print('generate velocity:', 'epoch =', epoch, 'loss =', l.item())

    loss_values.append(l.item())
    if l < l_min:
        l_min = l
        cnt = 0
    else:
        cnt = cnt + 1

    if cnt > 20:
        break

torch.save(net, 'net.pkl')
output = net(domain_drag_eval)
l = 0
domain_u_eval = torch.empty(set_num_eval, input_size, input_size)
domain_v_eval = torch.empty(set_num_eval, input_size, input_size)
domain_p_eval = torch.empty(set_num_eval, input_size, input_size)
for i in range(set_num_eval):
    for j in range(input_size):
        for k in range(input_size):
            if k == input_size - 1:
                domain_p_eval[i, j, k] = 1
            else:
                domain_p_eval[i, j, k] = output[i, j*(input_size - 1) + k]
            
            if j == 0 or j == input_size - 1 or k == 0 or k == input_size - 1:
                domain_v_eval[i, j, k] = 0
            else:
                domain_v_eval[i, j, k] = output[i, 2*(input_size - 1)*(input_size - 1) + (input_size - 2)*(j - 1) + k - 1]
            
            if not(j == 0) and not(j == input_size - 1) and not(k == 0):
                domain_u_eval[i, j, k] = output[i, (input_size)*(input_size - 1) + (input_size - 1)*(j - 1) + k - 1]
            elif k == 0:
                if j < 2:
                    domain_u_eval[i, j, k] = 1
                else:
                    domain_u_eval[i, j, k] = 0

for i in range(set_num_eval):
    for j in range(input_size):
        for k in range(input_size):
            if j == 0:
                domain_u_eval[i, j, k] = domain_u_eval[i, j + 1, k]
            elif j == input_size - 1:
                domain_u_eval[i, j, k] = domain_u_eval[i, j - 1, k]

for i in range(set_num_eval):
    l = l + loss(domain_u_eval[i], domain_v_eval[i], domain_p_eval[i], domain_drag_eval[i])

print(l.item())

#torch.save(domain_u_eval[0].detach(), 'u.pth')
#torch.save(domain_v_eval[0].detach(), 'v.pth')
# Convert tensors to NumPy arrays
p_array = domain_p_eval[0].detach().numpy()
u_array = domain_u_eval[0].detach().numpy()
v_array = domain_v_eval[0].detach().numpy()

# Creating a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 15))

# Plotting p
im0 = axs[0, 0].imshow(p_array, cmap='viridis')
axs[0, 0].set_title('p')
fig.colorbar(im0, ax=axs[0, 0])

# Plotting u
im1 = axs[1, 0].imshow(u_array, cmap='viridis')
axs[1, 0].set_title('u')
fig.colorbar(im1, ax=axs[1, 0])

# Plotting v
im2 = axs[1, 1].imshow(v_array, cmap='viridis')
axs[1, 1].set_title('v')
fig.colorbar(im2, ax=axs[1, 1])

# Define grid size and region
x = np.linspace(0, input_size - 1, input_size)
y = np.linspace(input_size - 1, 0, input_size)
X, Y = np.meshgrid(x, y)

# Plot quiver plot
im3 = axs[0, 1].quiver(X, Y, u_array, v_array, scale=10, label='velocity')
axs[0, 1].set_title('velocity')

plt.figure()
epoch_numbers = np.arange(1, len(loss_values) + 1)
relative_error = [x / 0.25 / 9 / 3 for x in loss_values]
plt.scatter(epoch_numbers, relative_error, color='blue', marker='o', label='Relative Error', s = 10)

# Add labels and title
plt.title('Relative Error Over Epochs')
plt.yscale('log')  # 将纵坐标设置为对数坐标
plt.xlabel('Epoch')
plt.ylabel('Relative Error')
plt.legend()
plt.grid(True)

# Display the figures
plt.show()