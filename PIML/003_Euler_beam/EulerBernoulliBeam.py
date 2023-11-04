#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:51:43 2023

@author: nguyenvanduc
"""

import tensorflow as tf
import datetime, os
#hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}
#0 (default) show all, 1 to filter out INFOR logs, 2 to additionally filter out
#WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.random.set_seed(1234)

print("TensorFlow version: {}".format(tf.__version__))


##############################################################################
# PINN #
# Generate a PINN of L hidden layers, each with n neurons
# Initialization: *Xavier*
# Activation: tanh(x)
class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):
        self.W = [] # Weight and biases
        self.parameters = 0 # Total number of parameters
        
        for i in range(len(layers)-1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            w = tf.Variable(w,trainable=True,name='w'+str(i+1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]),dtype='float64'),trainable=True,name='b'+str(i+1))
            self.W.append(w)
            self.W.append(b)
            self.parameters += input_dim * output_dim + output_dim
            
    def evaluate(self,x):
        # x = (x-lb)/(ub-lb)
        a = x
        for i in range(len(layers)-2):
            z = tf.add(tf.matmul(a,self.W[2*i]),self.W[2*i+1]) 
            a = tf.nn.tanh(z)
        a = tf.add(tf.matmul(a,self.W[-2]),self.W[-1]) 
        return a
    
    def get_weights(self):
        parameters_1d = []
        for i in range(len(layers)-1):
            w_1d = tf.reshape(self.W[2*i],[-1])
            b_1d = tf.reshape(self.W[2*i+1],[-1])
            parameters_1d = tf.concat([parameters_1d,w_1d],0)
            parameters_1d = tf.concat([parameters_1d,b_1d],0)
        return parameters_1d
    
    def set_weights(self, parameters):
        for i in range(len(layers)-1):
            shape_w = tf.shape(self.W[2*i]).numpy()
            size_w = tf.size(self.W[2*i]).numpy()
            shape_b = tf.shape(self.W[2*i+1]).numpy()
            size_b  = tf.size(self.W[2*i+1]).numpy()
            pick_w = parameters[0:size_w]
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) 
            parameters = np.delete(parameters, np.arange(size_w),0)
            pick_b = parameters[0:size_b]
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b))
            parameters = np.delete(parameters, np.arange(size_b),0)
            
    def loss_BC(self,x):
        x_lb = tf.Variable(x[0:1,:], dtype='float64', trainable=False)
        x_ub = tf.Variable(x[-1:,:], dtype='float64', trainable=False)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_lb)
            tape.watch(x_ub)
            u_lb = self.evaluate(x_lb)
            u_lb_x = tape.gradient(u_lb,x_lb)
            u_lb_xx = tape.gradient(u_lb_x,x_lb)
            u_ub = self.evaluate(x_ub)
            u_ub_x = tape.gradient(u_ub,x_ub)
            u_ub_xx = tape.gradient(u_ub_x,x_ub)    
        u_lb_xxx = tape.gradient(u_lb_xx,x_lb)
        u_ub_xxx = tape.gradient(u_ub_xx,x_ub)
        del tape
        loss_bc1 = x[0][1]*tf.reduce_mean(tf.square(u_lb)) + abs(x[0][1]-1)*tf.reduce_mean(tf.square(u_lb_xx))
        loss_bc2 = x[0][2]*tf.reduce_mean(tf.square(u_lb_x)) + abs(x[0][2]-1)*tf.reduce_mean(tf.square(u_lb_xxx))
        loss_bc3 = x[0][3]*tf.reduce_mean(tf.square(u_ub)) + abs(x[0][3]-1)*tf.reduce_mean(tf.square(u_ub_xx))
        loss_bc4 = x[0][4]*tf.reduce_mean(tf.square(u_ub_x)) + abs(x[0][4]-1)*tf.reduce_mean(tf.square(u_ub_xxx))
        return loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def loss_PDE(self, x_to_train_f):
        x_f = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)            
            z = self.evaluate(x_f)
            u_x = tape.gradient(z,x_f)
            u_xx = tape.gradient(u_x,x_f)
            u_xxx = tape.gradient(u_xx,x_f)
        u_xxxx = tape.gradient(u_xxx,x_f)
        del tape
        
        f = u_xxxx + 1 
        loss_f = tf.reduce_mean(tf.square(f))
        return loss_f
    
    def loss(self, x):
        loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss_BC(x)
        loss_pde = self.loss_PDE(x)
        return loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        
        BCi =np.random.randint(len(EBBC));
        BC = EBBC[BCi:BCi+1,:];
        BC = np.tile(BC, (len(x_train),1));
        x_i = np.hstack((x_train, BC));
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables) # trainable_variables is W (w+b)
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_i)
            loss_val = loss_pde + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
        grads = tape.gradient(loss_val, self.trainable_variables)
        del tape
        
        grads_1d = [] # flatten grads
        
        for i in range(len(layers)-1):
            grads_w_1d = tf.reshape(grads[2*i],[-1]) # flatten weights
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) # flatten biases
            
            grads_1d = tf.concat([grads_1d, grads_w_1d],0) #concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d],0) #concat grad_biases
            
        return loss_val.numpy(), grads_1d.numpy()
    
    def optimizer_callback(self, parameters):
        global counter
        if counter % 100 == 0:
            BCi =np.random.randint(len(EBBC));
            BC = EBBC[BCi:BCi+1,:];
            BC = np.tile(BC, (len(x_train),1));
            x_i = np.hstack((x_train, BC));
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_i)
            tf.print(counter, loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4)
        counter += 1
        
##############################################################################
# DATA PREP #
'''EBBC = np.array([[1, 1, 0, 0],
                 [0, 0, 1, 1],
                 [1, 1, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 1, 1],
                 [1, 0, 1, 0]]);'''
    
EBBC = np.array([[1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1]]);
x_train = np.arange(0,1.05,0.05)[:,None];

counter = 0


##############################################################################
# MODEL TRAINING AND TESTING #
layers = np.array([5,20,20,20,1])
PINN = Sequentialmodel(layers)
init_params = PINN.get_weights().numpy() 
start_time = time.time()
# train the model with Scipy L-BFGS optimizer
results = scipy.optimize.minimize(fun=PINN.optimizerfunc,
                                  x0 = init_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac=True, 
                                  callback=PINN.optimizer_callback,
                                  options={'disp':None,
                                          'maxcor': 200,
                                          'ftol': 1*np.finfo(float).eps,
                                          'gtol': 5e-8,
                                          'maxfun': 10000, 
                                          'maxiter': 5000, 
                                          'iprint': -1,
                                          'maxls': 20})

elapsed = time.time() - start_time
print('Training time: %.2f' %(elapsed))

PINN.set_weights(results.x)

BC = np.array([[1,1,1,1]]);
BC = np.tile(BC, (len(x_train),1));
x_i = np.hstack((x_train, BC));


y_i = PINN.evaluate(x_i)

fig, ax = plt.subplots()

ax.plot(x_i, y_i,'bo', linewidth=2.0)
ax.set(xlim=(-0.1, 1.1), xticks=np.arange(0, 1),
       ylim=(-0.0027, 0.0001), yticks=np.arange(-0.0027, 0.0001))
plt.show()

y_max = PINN.evaluate(np.array([[1/2, 1, 1, 1, 1]]))
#tf.print(y_max)









