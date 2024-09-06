"""
NS: 10*10离散化, 边条: u_in = const, p_out = const, 输出u, v, p, 由u, v, p计算loss
"""
import torch
import matplotlib.pyplot as plt
input_size = 10
u_in = 1
p_out = 1
rho = 1
nu = 1
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)

def l(u, v, p):
    grad_u_x = torch.zeros(input_size, input_size)
    grad_u_y = torch.zeros(input_size, input_size)
    grad_v_x = torch.zeros(input_size, input_size)
    grad_v_y = torch.zeros(input_size, input_size)
    grad_p_x = torch.zeros(input_size, input_size)
    grad_p_y = torch.zeros(input_size, input_size)
    grad_u_x_2 = torch.zeros(input_size, input_size)
    grad_u_y_2 = torch.zeros(input_size, input_size)
    grad_v_x_2 = torch.zeros(input_size, input_size)
    grad_v_y_2 = torch.zeros(input_size, input_size)

    for i in range(1, input_size - 1):
        for j in range(1, input_size - 1):
            grad_u_x[i, j] = (u[i, j + 1] - u[i, j - 1]) / 2
            grad_u_y[i, j] = (u[i - 1, j] - u[i + 1, j]) / 2
            grad_v_x[i, j] = (v[i, j + 1] - v[i, j - 1]) / 2
            grad_v_y[i, j] = (v[i - 1, j] - v[i + 1, j]) / 2
            grad_p_x[i, j] = (p[i, j + 1] - p[i, j - 1]) / 2
            grad_p_y[i, j] = (p[i - 1, j] - p[i + 1, j]) / 2
            grad_u_x_2[i, j] = u[i, j + 1] + u[i, j - 1] - 2*u[i, j]
            grad_u_y_2[i, j] = u[i + 1, j] + u[i - 1, j] - 2*u[i, j]
            grad_v_x_2[i, j] = v[i, j - 1] + v[i, j + 1] - 2*v[i, j]
            grad_v_y_2[i, j] = v[i - 1, j] + v[i + 1, j] - 2*v[i, j]

    #print(grad_u_x, grad_u_y, grad_v_x, grad_v_y, grad_p_x, grad_p_y, grad_u_x_2, grad_u_y_2, grad_v_x_2, grad_v_y_2)        
    e1 = torch.sum((u*grad_u_x + v*grad_u_y + 1/rho*(grad_p_x) - nu*(grad_u_x_2 + grad_u_y_2))**2)
    e2 = torch.sum((u*grad_v_x + v*grad_v_y + 1/rho*(grad_p_y) - nu*(grad_v_x_2 + grad_v_y_2))**2)
    e3 = torch.sum((grad_u_x + grad_v_y)**2)

    return (e1 + e2 + e3)


net = MLP()
opt = torch.optim.SGD(params = net.parameters(), lr = 0.0005)


for epoch in range(100000):
    domain_u = torch.empty(input_size, input_size)
    domain_v = torch.empty(input_size, input_size)
    domain_p = torch.empty(input_size, input_size)
    opt.zero_grad()
    for i in range(input_size):
        for j in range(input_size):
            input = torch.tensor([i + 1, j + 1]).float()
            output = net(input)
            if not(j == 0) and not(i == 0) and not(i == input_size - 1):
                domain_u[i, j] = output[0]
                domain_v[i, j] = output[1]
            
            if j == input_size - 1:
                domain_p[i, j] = p_out
            else:
                domain_p[i, j] = output[2]
    
    for i in range(input_size):
        for j in range(input_size):
            if j == 0:
                domain_u[i, j] = u_in
                domain_v[i, j] = 0
            elif i == 0:
                domain_u[i, j] = domain_u[i + 1, j]
                domain_v[i, j] = 0
            elif i == input_size - 1:
                domain_u[i, j] = 0
                domain_v[i, j] = 0

    training_loss = l(domain_u, domain_v, domain_p)        
    training_loss.backward()
    opt.step()

    print(epoch, 'loss =', training_loss.item())

domain_u = torch.zeros(input_size, input_size)
domain_v = torch.zeros(input_size, input_size)
domain_p = torch.zeros(input_size, input_size)
for i in range(input_size):
    for j in range(input_size):
        input = torch.tensor([i + 1, j + 1]).float()
        output = net(input)
        if not(j == 0) and not(i == 0) and not(i == input_size - 1):
            domain_u[i, j] = output[0]
            domain_v[i, j] = output[1]
        
        if j == input_size - 1:
            domain_p[i, j] = p_out
        else:
            domain_p[i, j] = output[2]
    
for i in range(input_size):
    for j in range(input_size):
        if j == 0:
            domain_u[i, j] = u_in
            domain_v[i, j] = 0
        elif i == 0:
            domain_u[i, j] = domain_u[i + 1, j]
            domain_v[i, j] = 0
        elif i == input_size - 1:
            domain_u[i, j] = 0
            domain_v[i, j] = 0

# Convert tensors to NumPy arrays
p_array = domain_p.detach().numpy()
u_array = domain_u.detach().numpy()
v_array = domain_v.detach().numpy()

# Creating a figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting p
im0 = axs[0].imshow(p_array, cmap='viridis')
axs[0].set_title('p')
fig.colorbar(im0, ax=axs[0])

# Plotting u
im1 = axs[1].imshow(u_array, cmap='viridis')
axs[1].set_title('u')
fig.colorbar(im1, ax=axs[1])

# Plotting v
im2 = axs[2].imshow(v_array, cmap='viridis')
axs[2].set_title('v')
fig.colorbar(im2, ax=axs[2])

# Display the figure
plt.show()




