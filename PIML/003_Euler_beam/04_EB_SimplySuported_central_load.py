#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:49:42 2023

@author: nguyenvanduc
"""

import deepxde as dde
import numpy as np
import tensorflow as tf
import torch

np.random.seed(1234)
tf.random.set_seed(1234)
torch.manual_seed(1234) # OK

def ddy(x,y):
    return dde.grad.hessian(y,x)

def dddy(x,y):
    return dde.grad.jacobian(ddy(x,y),x)

def pde(x,y):
    dy_xx = ddy(x,y)
    dy_xxxx = dde.grad.hessian(dy_xx,x)
    return dy_xxxx

def boundary_l(x,on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0],1)

def func(x):
    return  -(x ** 4) /24  + (5*x**3)/48 - (x**2)/16

geom = dde.geometry.Interval(0, 1)
bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x,y), boundary_l)

bc3 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x,y), boundary_r)


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
losshistory, train_state = model.train(iterations=5000,display_every=100)

model.save("Cantilever_DL_FES.h5")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)