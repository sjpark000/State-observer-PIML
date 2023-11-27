import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Nominal value of leader robot's speed
wr = 1
vr = 1

# Simulation time
tf = 10
ti = 0.01
tspan = np.arange(0,tf,ti)
sample_size = len(tspan)

# Observer gain
L = np.array([[1.7671], [3.2653], [5.0813]])
# Output matrix
C = np.array([1,0,0])
# Input matrix
B = np.array([[-1, 0],[0, 0],[0, -1]])

X = np.zeros((3,sample_size))
Xt = np.zeros((3,sample_size))
temp = np.zeros((1,sample_size))
error = np.zeros((1,sample_size))
X_hat = np.zeros((3,sample_size))
Xt_hat = np.zeros((3,sample_size))
err = np.zeros(((3,sample_size)))
DX = np.zeros((3,sample_size))
U = np.zeros((2,sample_size))


X[:,0] = np.array([-1,0,5])
Xt[:,0] = np.array([5,-5,0])
X_hat[:,0] = np.array([-1,0,0])
Xt_hat[:,0] = np.array([5,-5,0])
temp[:,0] = 2*(0.5 * np.sin(2*np.pi*(0+2)/6+1))+(0.5*np.cos(4*np.pi*(0+2)/6)+1)*1/np.sqrt((0+2)+1)

def plant(A, x, u):


    dx = A@x+B@u

    return dx

def plant_ob(A, x, e):

    dx = A@x+L@e

    return dx

def rk(A,x,t,u):

    k1 = plant(A, x, u) * t
    k2 = plant(A, x + k1 * 0.5, u) * t
    k3 = plant(A, x + k2 * 0.5, u) * t
    k4 = plant(A, x + k3, u) * t
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)

    return dx

def rk_ob(A,x,t,e):

    k1 = plant_ob(A, x, e) * t
    k2 = plant_ob(A, x + k1 * 0.5, e) * t
    k3 = plant_ob(A, x + k2 * 0.5, e) * t
    k4 = plant_ob(A, x + k3, e) * t
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)
    return dx



for i in range(sample_size-1):


    A = np.array([[0, 1, 0],
                  [-1, 0, 1],
                  [0, 0, 0]])

    # Given uncertainty for leader robot's speed
    Un = (0.5 * np.sin(2 * np.pi * (ti * i + 2) / 6 + 1) + 0.5 * np.cos(
        4 * np.pi * (ti * i + 2) / 6) + 1) * 1 / np.sqrt((ti * i + 2) + 1)
    #Un = 1-2*Un
    A_un = np.array([[0, wr+ Un, 0],
                  [-wr- Un, 0, vr + Un],
                  [0, 0, 0]])

    u = np.array([[(vr+Un)-1],[(vr+Un)-1]])
    error[:, i] = C @ (X[:, i] - X_hat[:, i])
    temp = rk(A_un,X[:,i].reshape(3,1),ti,u).reshape(3,1)
    X[:,i+1] = temp.reshape(1,3)

    X_hat[:,i+1] = rk_ob(A_un,X_hat[:,i],ti,error[:,i])


rk_ob = X_hat
rkt_ob = Xt_hat
rk = X
X_hat = X

# Hyperparameters
LR = 1e-3
MAX_EPOCH = 1000
BATCH_SIZE = 256
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)
SAVE_MODEL = False
PLOT = True


vectors1 = []
vectors2 = []

# Initial value for input data of neural network
for _ in range(np.size(tspan)):
    vectors1.append([-1, 0, 5])
    vectors2.append([-1, 0, 5])

vectors1_array = np.array(vectors1)
vectors2_array = np.array(vectors2)

x0 = vectors1_array[:,0]
x1 = vectors1_array[:,1]
x2 = vectors1_array[:,2]

xt0 = vectors2_array[:,0]
xt1 = vectors2_array[:,1]
xt2 = vectors2_array[:,2]

H = 128
# Neural Network(NN)
class NNfunc(nn.Module):
    def __init__(self):
        super(NNfunc, self).__init__()

        self.in_1 = nn.RNN(7, H, 3)
        self.in_2 = nn.RNN(H, H, 3)
        self.in_3 = nn.RNN(H, H, 3)
        self.in_4 = nn.Linear(H, H)
        self.in_5 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, H),
        )
        self.in_6 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, H),
        )
        self.in_7 = nn.Linear(H, 3)

        self.A1 = nn.Parameter(
            torch.tensor([[0,0.1,0],[-0.1,0,0.1],[0,0,0]]).to(DEVICE).to(torch.float32), requires_grad = False)
        self.BU = nn.Parameter(
            torch.tensor([0.3,0,0.1]).to(DEVICE).to(torch.float32), requires_grad = False
        )

        self.LC = nn.Parameter(torch.from_numpy(L @ C.reshape(1, 3)).to(DEVICE).to(torch.float32), requires_grad=False)

    def forward(self, input, is_train = False, y = None):

        x_hat1, states1 = self.in_1(input)
        x_hat2, states2 = self.in_2(x_hat1)
        x_hat3, states3 = self.in_3(x_hat1+x_hat2)
        x_hat4 = self.in_4(x_hat1+x_hat2+x_hat3)
        x_hat5 = self.in_5(x_hat4)
        x_hat6 = self.in_6(x_hat5)
        x_hat = self.in_7(x_hat6)
        dX_dt_gt = 0.

        if is_train:
            # State observer for PIML
            dX_dt_gt = ((self.A1 - self.LC) @ x_hat.T).T + self.BU + (self.LC @ y.T).T

        return x_hat, dX_dt_gt  # Estimated state and restored dynamics

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.eps = torch.tensor(np.sqrt(torch.finfo(torch.float32).eps))

    def loss_fn(self, outputs, targets):
        return nn.L1Loss(reduction="mean")(outputs, targets)

    # Training Function
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for X, y in data_loader:
            X = X.type(torch.float32).to(self.device)
            X.requires_grad=True
            y = y.type(torch.float32).to(self.device)

            X_hat, dX_dt_gt = self.model(X, is_train=True, y=y)
            X0, X1, X2 = X_hat[:, 0], X_hat[:, 1], X_hat[:, 2]

            dX0_dt = torch.autograd.grad(X0, X, torch.ones_like(X0), retain_graph=True, allow_unused=True)[0]
            dX1_dt = torch.autograd.grad(X1, X, torch.ones_like(X1), retain_graph=True, allow_unused=True)[0]
            dX2_dt = torch.autograd.grad(X2, X, torch.ones_like(X2), retain_graph=True, allow_unused=True)[0]

            dX_dt = torch.cat([dX0_dt[:,0].reshape(-1,1), dX1_dt[:,0].reshape(-1,1), dX2_dt[:,0].reshape(-1,1)], dim=1)

            self.optimizer.zero_grad()


            loss1 = self.loss_fn(X_hat, y)  # data loss
            loss2 = self.loss_fn(dX_dt_gt, dX_dt) # Physical equation loss function
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader), loss1

    # Evaluation Function
    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for X, y in data_loader:

            X = X.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)

            X_hat, _ =  self.model(X, False)
            loss = self.loss_fn(X_hat, y)
            final_loss += loss.item()
        return final_loss / len(data_loader)

y = X_hat


X = np.vstack((tspan,x0,x1,x2,rk_ob[0],rk_ob[1],rk_ob[2]))

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X.T, y.T, test_size=0.6
                                                                    , random_state=42))

train_dataloader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)

val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model = NNfunc().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
eng = Engine(model, optimizer, device=DEVICE)

best_loss = np.inf
early_stopping_iter = 10
early_stopping_counter = 0
e = np.arange(1,MAX_EPOCH+1)
loss = np.zeros(MAX_EPOCH)

# Training
for epoch in range(MAX_EPOCH):

    train_loss, loss1 = eng.train(data_loader=train_dataloader)
    loss[epoch] = train_loss

    val_loss = eng.evaluate(data_loader=val_dataloader)
    if epoch == 0 or (epoch+1)%5 == 0:
        print(f"Epoch: {epoch+1},\t Train Loss: {train_loss},\t Validation Loss: {val_loss}")
        print(f"Observer Error: {loss1}")

# Save the trained model
model_path = "model_one_data.pth"
torch.save(model.state_dict(), model_path)


if PLOT:
    plt.figure(1)
    y = X_hat

    X = np.vstack((tspan,x0,x1,x2, rk_ob[0], rk_ob[1], rk_ob[2]))


    plt.plot(tspan[None, :][0], y.T, label='True Solution')
    y_hat, _ = model(torch.from_numpy(X.T).to(DEVICE).to(torch.float32))
    y_hat = y_hat.cpu().detach()
    error = y_hat.numpy() - y.T

    plt.plot(tspan[None, :][0], y_hat, label='Estimated solution', linestyle = ':')

    plt.grid(True, which='both')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Estimated State Using State-Observer based PIML')
    plt.figure(2)
    plt.plot(e,loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function')

plt.show()