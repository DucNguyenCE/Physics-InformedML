import deepxde as dde
from deepxde.backend import tf
import numpy as np
import scipy.io

def gen_testdata():
    data = scipy.io.loadmat("../000_Data/001_burgers_shock_mu_01_pi.mat")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def pde(x,y):
    dy_x = dde.grad.jacobian(y, x,i=0,j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01/np.pi * dy_xx

geom = dde.geometry.Interval(-1, 1) # Define an interval representing the range [-1, 1]
timedomain = dde.geometry.TimeDomain(0, 0.99) # Define a one-dimensional time domain with a spatial domain of [0, 0.99]
geomtime = dde.geometry.GeometryXTime(geom, timedomain) # Define a two-dimensional time-varying system with the spatial domain


bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _,on_boundary: on_boundary) # Dirichlet boundary conditions: y(x) = func(x). func(x) = 0
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:,0:1]),lambda _, on_initial: on_initial) # Initial conditions: y([x, t0]) = func([x, t0]).

data = dde.data.TimePDE(geomtime, pde, [bc,ic],
                        num_domain = 2540, num_boundary=80, num_initial=160) 
# Time-dependent fractional PDE solver.
# num_domain = 2540: number of training residual points sampled inside the domain.
# num_boundary = 80: number of training points sampled on the boundary.
# num_initial = 160: number of initial residual points for the initial conditions.


net = dde.nn.FNN([2] + [20]*3 +[1],"tanh", "Glorot normal") # Feedforward neural network
# The Glorot normal initializer is a method for initializing the weights of a neural network
# It is named after Xavier Glorot.

model = dde.Model(data, net)
model.compile("adam",lr=1e-3)

losshistory, train_state = model.train(iterations=15000)

model.compile("L-BFGS-B")
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))