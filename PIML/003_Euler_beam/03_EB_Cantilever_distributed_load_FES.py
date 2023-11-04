#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:50:22 2022

@author: nguyenvanduc
"""

import deepxde as dde
import numpy as np
import tensorflow as tf
import torch

np.random.seed(1234)
tf.random.set_seed(1234)
torch.manual_seed(1234) # OK
K = 10

def ddy(x,y):
    return dde.grad.hessian(y,x)

def dddy(x,y):
    return dde.grad.jacobian(ddy(x,y),x)

def pde(x,y):
    dy_xx = ddy(x,y)
    dy_xxxx = dde.grad.hessian(dy_xx,x)
    return dy_xxxx + K

def boundary_l(x,on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0],1)

def func(x):
    return K* ( -(x ** 4) /24  + (5*x**3)/48 - (x**2)/16)

geom = dde.geometry.Interval(0, 1)
bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l) # w = 0
bc2 = dde.icbc.NeumannBC(geom, lambda x:0, boundary_l)    # w'= 0
bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x,y), boundary_r) # M = 0
bc4 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_r) # w = 0

data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain = 20,
    num_boundary=2,
    solution = func,
    num_test=100)

layer_size = [1] + [20]*3 + [1]
activation ="tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000,display_every=100)

model.save("Cantilever_DL_FES.h5")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Y_max = model.predict(5/8)

print("Y_max = ", Y_max)