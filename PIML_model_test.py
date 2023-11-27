import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 저장된 모델을 불러올 경로와 파일 이름 지정
model_path = "model_one_data_init_1.pth"

#L = np.array([[6.], [3.5], [1.5]])
#L = np.array([[1.7671], [3.2653], [5.0813]])
L = np.array([[6],[24],[25]])
#L = np.array([[1.1055], [2.8342], [1.1872]])
C = np.array([1,0,0])
B = np.array([[-1, 0],[0, 0],[0, -1]])
U1 = np.array([[3],[1]])
wr = 1
vr = 1
tf = 10
ti = 0.01
tspan = np.arange(0,tf,ti)
sample_size = len(tspan)
H = 128

# Neuronal Network
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

        self.LC = nn.Parameter(torch.from_numpy(L @ C.reshape(1, 3)).to(DEVICE).to(torch.float32), requires_grad=False)

        self.BU = nn.Parameter(
            torch.tensor([0.3,0,0.1]).to(DEVICE).to(torch.float32), requires_grad = False
        )

    def forward(self, input, is_train = False, y = None):

        x_hat1, states1 = self.in_1(input)
        x_hat2, states2 = self.in_2(x_hat1)
        x_hat3, states3 = self.in_3(x_hat1+x_hat2)
        x_hat4 = self.in_4(x_hat1+x_hat2+x_hat3)
        x_hat5 = self.in_5(x_hat4)
        x_hat6 = self.in_6(x_hat5)
        x_hat = self.in_7(x_hat6)
        dX_dt_gt = 0.

        x_hat_np = x_hat.cpu()

        dX_dt_gt = A_un @ x_hat_np.numpy().T + B @ U + L @ C.reshape(1,3) @ (X_hat-x_hat_np.numpy().T)

        return x_hat, dX_dt_gt



X = np.zeros((3,sample_size))
Xt = np.zeros((3,sample_size))
temp = np.zeros((1,sample_size))
error = np.zeros((1,sample_size))
X_hat = np.zeros((3,sample_size))
Xt_hat = np.zeros((3,sample_size))
err = np.zeros(((3,sample_size)))
DX = np.zeros((3,sample_size))
U = np.zeros((2,sample_size))

#X[:,0] = np.array([-1,0,0])
X[:,0] = np.array([1,0,5])
Xt[:,0] = np.array([5,-5,0])
X_hat[:,0] = np.array([1,0,0])
Xt_hat[:,0] = np.array([5,-5,0])
temp[:,0] = 2*(0.5 * np.sin(2*np.pi*(0+2)/6+1))+(0.5*np.cos(4*np.pi*(0+2)/6)+1)*1/np.sqrt((0+2)+1)
#X[:,0] = np.array([np.sqrt(2),0,0])

def plant(A, x, u):


    dx = A@x+B@u

    return dx

def plant_ob(A, x, u, e):

    dx = A@x+B@u+(L@e).reshape(3,1)

    return dx

def rk(A,x,t,u):

    k1 = plant(A, x, u) * t
    k2 = plant(A, x + k1 * 0.5, u) * t
    k3 = plant(A, x + k2 * 0.5, u) * t
    k4 = plant(A, x + k3, u) * t
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)

    return dx

def rk_ob(A,x,t,u,e):

    k1 = plant_ob(A, x, u, e) * t
    k2 = plant_ob(A, x + k1 * 0.5, u, e) * t
    k3 = plant_ob(A, x + k2 * 0.5, u, e) * t
    k4 = plant_ob(A, x + k3, u, e) * t
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)
    return dx



for i in range(sample_size-1):

    # A = np.array([[0, wr+ 0.4*np.sin((2*i+1)*np.pi/100), 0],
    #              [-wr- 0.4*np.sin((2*i+1)*np.pi/100), 0, vr + 0.4*np.sin((2*i+1)*np.pi/100)],
    #              [0, 0, 0]])
    A = np.array([[0, 1, 0],
                  [-1, 0, 1],
                  [0, 0, 0]])

    Un = (0.5 * np.cos(2 * np.pi * (ti * i + 2) / 6 + 1) + 0.5 * np.cos(
        4 * np.pi * (ti * i + 2) / 6) + 1) * 1 / np.sqrt((ti * i + 2) + 1)
    #Un = 1-2*Un
    #Un = 0.7-Un
    A_un = np.array([[0, wr+ Un, 0],
                  [-wr- Un, 0, vr + Un],
                  [0, 0, 0]])

    u = np.array([[1-(vr+Un)], [1-(wr-Un)]])
    #u = np.array([[0],[0]])
    #u = np.array([[vr+Un-1],[wr+Un-1]])
    error[:, i] = C @ (X[:, i] - X_hat[:, i])
    temp = rk(A_un,X[:,i].reshape(3,1),ti,u).reshape(3,1)
    X[:,i+1] = temp.reshape(1,3)

    temp_ob = rk_ob(A,X_hat[:,i].reshape(3,1),ti,u,error[:,i]).reshape(3,1)
    X_hat[:, i + 1] = temp_ob.reshape(1,3)

    U[:,i] = u.reshape(1,2)
    #temp[:,i+1] = (0.5 * np.sin(2*np.pi*(ti*i+2)/6+1)+0.5*np.cos(4*np.pi*(ti*i+2)/6)+1)*1/np.sqrt((ti*i+2)+1)

# 초기 값
# 1000개의 3차원 벡터를 담을 리스트 생성
vectors1 = []
vectors2 = []

# 1000개의 벡터를 생성하여 리스트에 추가
for _ in range(np.size(tspan)):
    # x = np.random.randint(-1, 2)   # x 좌표
    # y = np.random.randint(-1, 2)   # y 좌표
    # z = np.random.randint(-1, 2)   # z 좌표
    # vectors.append([x, y, z])
    vectors1.append([1, 0, 5])


vectors1_array = np.array(vectors1)

x0 = vectors1_array[:,0]
x1 = vectors1_array[:,1]
x2 = vectors1_array[:,2]


# 모델 아키텍처 정의
model = NNfunc().to(DEVICE)

# 모델의 가중치와 편향 불러오기
model.load_state_dict(torch.load(model_path))

y = np.vstack((tspan, x0, x1, x2, X_hat[0], X_hat[1], X_hat[2]))

# 테스트 데이터를 PyTorch Tensor로 변환하고 GPU에 전달 (필요 시)
X_test_tensor = torch.tensor(y.T, dtype=torch.float32).to(DEVICE)

# 모델에 테스트 데이터 전달하여 예측 수행
with torch.no_grad():
    y_pred, grad = model(X_test_tensor, is_train=False)  # is_train=False로 설정하여 예측 모드로 전환

# 예측 결과를 CPU로 이동하고 NumPy 배열로 변환
y_pred = y_pred.cpu().numpy()

gradient0 = np.gradient(X_hat[0])*100
gradient1 = np.gradient(X_hat[1])*100
gradient2 = np.gradient(X_hat[2])*100

x_dot = np.gradient(X[0])*100
y_dot = np.gradient(X[1])*100
theta_dot = np.gradient(X[2])*100

dx = np.vstack([gradient0, gradient1, gradient2])

np.savetxt('E:/PINN/different_example/gradient[0].csv', gradient0, delimiter=",")
np.savetxt('E:/PINN/different_example/gradient[1].csv', gradient1, delimiter=",")
np.savetxt('E:/PINN/different_example/gradient[2].csv', gradient2, delimiter=",")
np.savetxt('E:/PINN/different_example/y_hat2[0].csv', X_hat[0], delimiter=",")
np.savetxt('E:/PINN/different_example/y_hat2[1].csv', X_hat[1], delimiter=",")
np.savetxt('E:/PINN/different_example/y_hat2[2].csv', X_hat[2], delimiter=",")

np.savetxt('E:/PINN/different_example/rk[0].csv', X[0], delimiter=",")
np.savetxt('E:/PINN/different_example/rk[1].csv', X[1], delimiter=",")
np.savetxt('E:/PINN/different_example/rk[2].csv', X[2], delimiter=",")

np.savetxt('E:/PINN/different_example/x_dot.csv', x_dot, delimiter=",")
np.savetxt('E:/PINN/different_example/y_dot.csv', y_dot, delimiter=",")
np.savetxt('E:/PINN/different_example/theta_dot.csv', theta_dot, delimiter=",")

plt.figure(1)

plt.plot(tspan[:,None], X.T, label = 'True Solution')
plt.plot(tspan, X_hat[0], linestyle = ':', label = 'Estimated x_e')
plt.plot(tspan, X_hat[1], linestyle = ':', label = 'Estimated y_e')
plt.plot(tspan, X_hat[2], linestyle = ':', label = 'Estimated theta_e')
plt.xlabel('Time')
plt.ylabel('State')

plt.grid(True, which='both')
plt.legend()

plt.show()