clear all
m = 3;
n = 0.5;

A1 = [0 1 0;-1 0 m;0 0 0];
A2 = [0 1 0;-1 0 n;0 0 0];
A = [0 1 0;-1 0 1;0 0 0];
B = [-1 0;0 0;0 -1];
C = [1 0 0];
M = [1 0;0 1;0 0];
N = [0 1 0;-1 0 1];
%M = zeros(3);
%N = zeros(3);

Q = sdpvar(3,3,'symmetric');
R = sdpvar(3,3,'symmetric');

L = sdpvar(3,1);

lo = sdpvar(1,1);
Y = sdpvar(2,3);
Y1 = sdpvar(2,3);
Y2 = sdpvar(2,3);
gamma = sdpvar(1,1);

epsilon1 = sdpvar(1,1);
epsilon2 = sdpvar(1,1);

%Y1 = Y;
%Y2 = Y;

F1 = [Q>0, lo>0, epsilon1>0, epsilon2>0, R>0];

X11 = [Q*A'+A'*Q+B*Y2+Y2'*B' M+B Q*N'+(Y1-Y2)' N';...
    (M+B)' -epsilon1*eye(2) zeros(2) zeros(2);...
    (Q*N'+(Y1-Y2)')' zeros(2) -epsilon1*eye(2) zeros(2);...
    N zeros(2) zeros(2) -epsilon2*eye(2)]
X22 = [A'*R+R*A-L*C-C'*L' R*M;...
    (R*M)' -epsilon2*eye(2)]
%F2 = [[X11 zeros(3) P*M zeros(3);
%    zeros(3) X22 zeros(3) R*M;
%    M'*P zeros(3) -epsilon1*eye(3) zeros(3);
%    zeros(3) M'*R zeros(3) -epsilon2*eye(3)]<-lo*eye(12)];
F2 = [[X11 zeros(9,5);zeros(5,9) X22]<-lo*eye(14)];

F3 = [[Q [C*Q]';C*Q C*Q*C']>0]
F4 = [trace(C*Q*C')<gamma];
F5 = [[A1*Q+B*Y1+Q*A1'+Y1'*B']<0];
F6 = [[A2*Q+B*Y2+Q*A2'+Y2'*B']<0];

F = [F1,F2,F3,F4,F5,F6];
solvesdp(F,[], sdpsettings('solver', 'sedumi'));
checkset(F);

%lo = double(lo);

L = double(L)
Y1 = double(Y1);
Y2 = double(Y2);
Q = double(Q);
K1 = Y1*inv(Q)
K2 = Y2*inv(Q)

H2_norm = double(gamma)

P = inv(Q);
R = double(R);
S = sdpvar(6,6,'symmetric');
FF = [[P zeros(3);zeros(3) R]*[M*0.5*N zeros(3);zeros(3) A-L*C]+[M*0.5*N zeros(3);zeros(3) A-L*C]'*[P zeros(3);zeros(3) R]'<=-S*eye(6), S>0];
solvesdp(FF,[], sdpsettings('solver', 'sedumi'));
checkset(FF);
S = double(S);
max_bound = min(eig(S))/(2*max(eig([P zeros(3);zeros(3) R])))
%K = inv(P)*Y;