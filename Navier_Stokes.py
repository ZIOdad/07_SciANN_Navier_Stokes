import os
print(__file__)
print(os.getcwd())
print(os.path.dirname(os.path.realpath(__file__)))

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import scipy.io

#training data
def PrepareData(num_data=5000, random=True):

    # Get data file from:
    #         https://github.com/maziarraissi/PINNs/tree/master/main/Data/cylinder_nektar_wake.mat
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')

    print(data.keys()) #dict_keys(['__header__', '__version__', '__globals__', 'X_star', 't', 'U_star', 'p_star'])

    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    # print(type(X_star)) # array
    # print(X_star.shape) # (5000, 2)
    N = X_star.shape[0] # 5000
    T = t_star.shape[0] # 200

    # Rearrange Data
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    # print(type(XX)) # array
    # print(XX.shape) # (5000, 200)

    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T

    # Pick random data.
    if random: #함수 입력인자(True)
        idx = np.random.choice(N*T, num_data, replace=False) #비복원추출(동일한 원소를 뽑을 수 없음)
    else:
        idx = np.arange(0, N*T)

    x = XX.flatten()[idx,None] # NT x 1
    y = YY.flatten()[idx,None] # NT x 1
    t = TT.flatten()[idx,None] # NT x 1

    u = UU.flatten()[idx,None] # NT x 1
    v = VV.flatten()[idx,None] # NT x 1
    p = PP.flatten()[idx,None] # NT x 1

    return (x,y,t,u,v,p)

x_train, y_train, t_train, u_train, v_train, p_train = PrepareData(500, random=True)

"""
PINN setup
"""
#input variable
x = sn.Variable("x", dtype='float64')
y = sn.Variable("y", dtype='float64')
t = sn.Variable("t", dtype='float64')

#solution variable
P = sn.Functional("P", [x, y, t], 8*[20], 'tanh')
Psi = sn.Functional("Psi", [x, y, t], 8*[20], 'tanh')

#parameters
lambda1 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda1")
lambda2 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda2")

#Use sn.diff and other mathematical operations to set up the PINN model.
u = sn.diff(Psi, y)
v = -sn.diff(Psi, x)

u_t = sn.diff(u, t)
u_x = sn.diff(u, x)
u_y = sn.diff(u, y)
u_xx = sn.diff(u, x, order=2)
u_yy = sn.diff(u, y, order=2)

v_t = sn.diff(v, t)
v_x = sn.diff(v, x)
v_y = sn.diff(v, y)
v_xx = sn.diff(v, x, order=2)
v_yy = sn.diff(v, y, order=2)

p_x = sn.diff(P, x)
p_y = sn.diff(P, y)

#Define losses
d1 = sn.Data(u)
d2 = sn.Data(v)
d3 = sn.Data(P)

c1 = sn.Tie(-p_x, u_t+lambda1*(u*u_x+v*u_y)-lambda2*(u_xx+u_yy))
c2 = sn.Tie(-p_y, v_t+lambda1*(u*v_x+v*v_y)-lambda2*(v_xx+v_yy))
c3 = sn.Data(u_x + v_y)

c4 = Psi*0.0

#Define the optimization model
model = sn.SciModel(
    inputs=[x, y, t],
    targets=[d1, d2, d3, c1, c2, c3, c4],
    loss_func="mse"
)

input_data = [x_train, y_train, t_train]
data_d1 = u_train
data_d2 = v_train
data_d3 = p_train
data_c1 = 'zeros'
data_c2 = 'zeros'
data_c3 = 'zeros'
data_c4 = 'zeros'
target_data = [data_d1, data_d2, data_d3, data_c1, data_c2, data_c3, data_c4]

history = model.train(
    x_true=input_data,
    y_true=target_data,
    epochs=5000,
    batch_size=10,
    shuffle=True,
    learning_rate=0.001,
    reduce_lr_after=100,
    stop_loss_value=1e-8,
    verbose=1
)

model.save_weights('trained-navier-stokes.hdf5')

print("lambda1: {},  lambda2: {}".format(lambda1.value, lambda2.value))

fig1 = plt.figure(figsize=(6,5))
plt.semilogy(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot()

model.load_weights('trained-navier-stokes.hdf5')

x_test, y_test, t_test, u_test, v_test, p_test = PrepareData(5000, random=False)
print("----test-----")
print(x_test.shape)
print(type(x_test))

x_pred = x_test
y_pred = y_test
t_pred = t_test
f_pred = P.eval(model, [x_pred, y_pred, t_pred])
print(f_pred.shape)
f_pred = f_pred.reshape(50, 100, 200)
print(f_pred.shape)
f_pred = f_pred[:,:,99:100]
f_pred = f_pred.reshape(50, 100)
print(f_pred.shape)

x_test = x_test.reshape(50, 100, 200)[:,:,99:100]
y_test = y_test.reshape(50, 100, 200)[:,:,99:100]
p_test = p_test.reshape(50, 100, 200)[:,:,99:100]
print(x_test.shape)
x_test = x_test.reshape(50, 100)
y_test = y_test.reshape(50, 100)
p_test = p_test.reshape(50, 100)
print(x_test.shape)

fig2 = plt.figure(figsize=(3,4))
plt.pcolor(x_test, y_test, p_test, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

fig3 = plt.figure(figsize=(3,4))
plt.pcolor(x_test, y_test, f_pred, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

plt.show()
