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
model_path = "trained_PIML.pth"

C = np.array([1,0,0])
B = np.array([[-1, 0],[0, 0],[0, -1]])
L = np.array([[6],[24],[25]])

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

    Un = (0.5 * np.sin(2 * np.pi * (ti * i + 2) / 6 + 1) + 0.5 * np.cos(
        4 * np.pi * (ti * i + 2) / 6) + 1) * 1 / np.sqrt((ti * i + 2) + 1)
    #Un = 1-2*Un
    #Un = 0.7-Un
    A_un = np.array([[0, wr+ Un, 0],
                  [-wr- Un, 0, vr + Un],
                  [0, 0, 0]])
    #u = np.array([[0],[0]])
    u = np.array([[1-(vr+Un)], [1-(wr+Un)]])

    error[:, i] = C @ (X[:, i] - X_hat[:, i])
    temp = rk(A_un,X[:,i].reshape(3,1),ti,u).reshape(3,1)
    X[:,i+1] = temp.reshape(1,3)

    temp_ob = rk_ob(A_un,X_hat[:,i].reshape(3,1),ti,u,error[:,i]).reshape(3,1)
    X_hat[:, i + 1] = temp_ob.reshape(1,3)

    U[:,i] = u.reshape(1,2)


# 초기 값
# 1000개의 3차원 벡터를 담을 리스트 생성
vectors1 = []
vectors2 = []

# 1000개의 벡터를 생성하여 리스트에 추가
for _ in range(np.size(tspan)):

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


m = 3
n = 0.5
A1 = np.array([[0, m, 0],[-m, 0, m],[0, 0, 0]])
A2 = np.array([[0, n, 0],[-n, 0, n],[0, 0, 0]])
tf = 10
ti = 0.01
ti_new = 0.1
tspan = np.arange(0,tf,ti)
tspan_new = np.arange(0,tf,ti_new)
sample_size = len(tspan)

dx = np.vstack([gradient0, gradient1, gradient2])
# print(A1@dx[:,0].reshape(2,1))

x_hat = np.vstack([x0, x1, x2])
x_hat_piml = np.vstack([X_hat[0], X_hat[1], X_hat[2]])
alpha = np.arange(0,1,0.001)
norm1 = np.arange(0,1,0.001).tolist()
norm2 = np.arange(0,1,0.001).tolist()
value1 = np.zeros((3, len(alpha)))
value2 = np.zeros((3, len(alpha)))
final1 = np.arange(0,tf,ti)
final2 = np.arange(0,tf,ti)
final3 = []

real_alpha = np.arange(0,tf,ti)

aa = np.size(tspan)-2

for i in range(np.size(tspan)):

    for j in range(np.size(alpha)):

        if i == np.size(tspan)-1:

            value1[:, j] = alpha[j] * (A1 - A2) @ (x_hat[:,i-1] + x_hat[:,i]) + A2 @ (x_hat[:,i-1] + x_hat[:,i]) - (dx[:,i-1] + dx[:,i])
            value2[:, j] = alpha[j] * (A1 - A2) @ (x_hat_piml[:, i-1] + x_hat_piml[:, i]) + A2 @ (x_hat_piml[:, i-1] + x_hat_piml[:, i]) - (dx[:, i-1] + dx[:, i])

            norm1[j] = value1[:,j][0] ** 2 + value1[:,j][1] ** 2 + value1[:,j][2] ** 2
            norm2[j] = value2[:,j][0] ** 2 + value2[:,j][1] ** 2 + value2[:,j][2] ** 2
        else:
            value1[:, j] = alpha[j] * (A1 - A2) @ (x_hat[:,i] + x_hat[:,i + 1]) + A2 @ (x_hat[:,i] + x_hat[:,i + 1]) - (dx[:,i] + dx[:,i + 1])
            value2[:, j] = alpha[j] * (A1 - A2) @ (x_hat_piml[:, i] + x_hat_piml[:, i + 1]) + A2 @ (
                        x_hat_piml[:, i] + x_hat_piml[:, i + 1]) - (dx[:, i] + dx[:, i + 1])

            norm1[j] = value1[:,j][0] ** 2 + value1[:,j][1] ** 2 + value1[:,j][2] ** 2
            norm2[j] = value2[:, j][0] ** 2 + value2[:, j][1] ** 2 + value2[:, j][2] ** 2

    final1[i] = alpha[norm1.index(min(norm1))]
    final2[i] = alpha[norm2.index(min(norm2))]

    if (i % 10 == 0):
        final3.append(final2[i])


for i in range(np.size(tspan)):

    Un = (0.5 * np.sin(2 * np.pi * (ti * i + 2) / 6 + 1) + 0.5 * np.cos(
        4 * np.pi * (ti * i + 2) / 6) + 1) * 1 / np.sqrt((ti * i + 2) + 1)

    real_alpha[i] = (1-n+(Un))/(m-n)


mse = []
mse1 = []

for i in range(100):
    mse.append(real_alpha[i])
    mse1.append(final2[i])

from sklearn.metrics import mean_squared_error
M = mean_squared_error(mse, mse1)

print("Parameter Error(MSE): ", M)

tf_new = 20
ti_new = 0.01
tspan_new = np.arange(0,tf_new,ti_new)
sample_size = len(tspan_new)

X = np.zeros((3, sample_size))

Leader = np.zeros((3, sample_size))
Follower = np.zeros((3, sample_size))

U_leader = np.zeros((2, sample_size))
U_follower = np.zeros((2, sample_size))
U_e = np.zeros((2, sample_size))

Leader[:,0] = np.array([1, 0, 0])
Follower[:,0] = np.array([0, 0, 0])


X[:,0] = Leader[:,0]-Follower[:,0]


alpha = final2[0:sample_size]
alpha_true = real_alpha[0:sample_size]

# K1 = np.array([[0.9733,0.1368,0.1110],[-0.8583,1.3540,0.8570]]) # Best Solution
# K2 = np.array([[0.7778,-0.1246,0.0085],[-0.0992,0.2957,0.5439]]) # Best Solution

K1 = np.array([[2.6667,-1.6741,-2.6579],[0.4851,2.0094,2.1071]]) # Best Solution
K2 = np.array([[3.5128,1.0083,1.0249],[2.5554,3.4222,5.6808]]) # Best Solution

# K_constant = np.array([[0.9106,-0.2211,-0.0829],[-0.3281,0.5162,0.9646]])
K_constant = np.array([[3.0898,-0.3329,-0.8165],[1.5202,2.7158,3.8939]])


def Eplant(A, x, u):

    dx = A@x+B@u

    return dx


def LFplant(x, u):

    dx1 = u[0] * np.cos(x[2])
    dx2 = u[0] * np.sin(x[2])
    dx3 = u[1]
    dx = np.array([dx1, dx2, dx3])
    return dx


def rke(A, x, u, t):
    k1 = Eplant(A, x, u) * t
    k2 = Eplant(A, x + k1 * 0.5, u) * t
    k3 = Eplant(A, x + k2 * 0.5, u) * t
    k4 = Eplant(A, x + k3, u) * t
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)
    return dx


def rkLF(x, u, t):

    k1 = LFplant(x, u) * t
    k1 = k1.reshape(1,3)[0]
    k2 = LFplant(x + k1 * 0.5, u) * t
    k2 = k2.reshape(1, 3)[0]
    k3 = LFplant(x + k2 * 0.5, u) * t
    k3 = k3.reshape(1, 3)[0]
    k4 = LFplant(x + k3, u) * t
    k4 = k4.reshape(1, 3)[0]
    dx = x + ((k1 + k4) / 6 + (k2 + k3) / 3)
    return dx


m = 3
n = 0.5

B = np.array([[-1, 0], [0, 0], [0, -1]])
Un = np.zeros((1,sample_size))
Un1 = np.zeros((1,sample_size))
Un2 = np.zeros((1,sample_size))

theta_dot_new = []

for i in range(sample_size-1):

    Un[0][i] = (0.5 * np.sin(2 * np.pi * (ti * i + 2) / 6 + 1) + 0.5 * np.cos(
        4 * np.pi * (ti * i + 2) / 6) + 1) * 1 / np.sqrt((ti * i + 2) + 1)
    #Un[0][i] = 0.7-Un[0][i]
    Un1[0][i] = 1+Un[0][i] # 실제 부여하는 리더의 선 속도
    Un2[0][i] = 1-Un[0][i] # 실제 부여하는 리더의 각 속도

    # if (i % 10 == 0):
    #     theta_dot_new.append(1+theta_dot[i])

    if  i<1000 :
        vr_true = Un1[0][i]
        wr_true = Un2[0][i]
        vr = 0 + (m * alpha[i] + n * (1 - alpha[i])) # 제안하는 방법으로부터 추정된 리더의 선 속도
        #vr = x_dot[i]+1-y[i]
        #vr = (y_dot[i]+x[i])/theta_dot[i]
        wr = 1 + theta_dot[i] # 제안하는 방법으로부터 추정된 리더의 각 속도
        K = alpha[i] * K1 + (1 - alpha[i]) * K2
        UL = np.array([vr_true, wr_true])
        UF = K @ X[:, i]+np.array([vr,wr_true])
        #UF = K @ X[:, i]
        #U = UF - np.array([Un1[0][i], Un2[0][i]])
        U = UF - UL

    else:
        vr_true = 0
        wr_true = 0
        vr = 0
        wr = 0
        UL = np.array([vr_true, wr_true])
        UF = K_constant @ X[:, i] + np.array([vr,wr])
        U = UF

    # 제어 입력

    #UF = K_constant @ X[:, i]



    A = np.array([[0, UF[1], 0],
                    [-UF[1], 0, vr],
                    [0, 0, 0]])

    # 에러 다이나믹스
    X[:, i + 1] = rke(A, X[:, i], U, ti)

    U_leader[:,i] = np.array([vr_true,wr_true])
    U_follower[:, i] = UF
    U_e[:,i] = UF - np.array([Un1[0][i], Un2[0][i]])
    Leader[:, i + 1] = rkLF(Leader[:, i], UL, ti)
    Follower[:, i + 1] = rkLF(Follower[:, i], UF, ti)

    #X[:,i+1]=Leader[:,i+1]-Follower[:,i+1]

vr = U_leader[0,:999]
wr = U_leader[1,:999]
vf = U_follower[0,:999]
wf = U_follower[1,:999]


mse = []
mse1 = []


def calculate_mse(vector_list1, vector_list2):
    if len(vector_list1) != len(vector_list2):
        raise ValueError("Vector lists must have the same length")

    squared_errors = []
    for vec1, vec2 in zip(vector_list1, vector_list2):
        squared_error = sum([(x - y)**2 for x, y in zip(vec1, vec2)])
        squared_errors.append(squared_error)

    mse = sum(squared_errors) / len(vector_list1)
    return mse

for i in range(1000):
    mse.append((Leader[0,i],Leader[1,i]))

    mse1.append((Follower[0,i], Follower[1,i]))


mse1 = calculate_mse(mse,mse1)
print(mse1)

plt.figure(1)
plt.plot(tspan,final2, label='State observer based PIML method', linestyle="--", linewidth=2)
plt.plot(tspan,real_alpha, label='Numerical method')
plt.title("Estimated parameter")
plt.xlabel("Time")
plt.grid(True, which='both')
plt.legend()

plt.figure(2)
plt.plot(Leader[0,:],Leader[1,:], label = 'Leader',linewidth=2)
plt.plot(Follower[0,:],Follower[1,:], label = 'Follower', linestyle="--", linewidth=2)
plt.grid(True, which='both')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Leader-Follower Trajectory")
plt.legend()

plt.show()