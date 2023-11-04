#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:46:51 2023

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
# Some information
q = -10
E = 3e7
I = 1/12
L = 100

nu = 0.3
G = E/2/(1+nu)
kapa = 5/6
A = 1
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
        self.lagrange_1 = tf.Variable(tf.cast(tf.ones([4,1]), dtype = 'float64'), trainable = True) # 4 boundary points
        self.lagrange_2 = tf.Variable(tf.cast(tf.ones([26,2]), dtype = 'float64'), trainable = True) # 26 train points

    
    def evaluate(self,x):
        x = (x-lb)/(ub-lb)
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
            
    def loss_BC(self,lb,ub):
        x_lb = tf.Variable(lb, dtype='float64', trainable=False)
        x_ub = tf.Variable(ub, dtype='float64', trainable=False)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_lb)
            tape.watch(x_ub)
            u_lb = self.evaluate(x_lb)
            u_ub = self.evaluate(x_ub)            
        del tape
        loss_bc1 = tf.reduce_mean(tf.square(self.lagrange_1[0]*u_lb[:,0:1]))
        loss_bc2 = tf.reduce_mean(tf.square(self.lagrange_1[1]*u_lb[:,1:2]))
        loss_bc3 = tf.reduce_mean(tf.square(self.lagrange_1[2]*u_ub[:,0:1]))
        loss_bc4 = tf.reduce_mean(tf.square(self.lagrange_1[3]*u_ub[:,1:2]))
        return loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        
        x_f = g[:,0:1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)            
            z = self.evaluate(x_f)
            z1 = z[:,0:1]
            z2 = z[:,1:2]

            phi_x = tape.gradient(z2,x_f)
            phi_xx = tape.gradient(phi_x,x_f)
        
        u_x = tape.gradient(z1,x_f)
        phi_xxx = tape.gradient(phi_xx,x_f)
        # Computes the gradient using operations recorded in context of this tape.
        del tape
        
        f1 = E*I*phi_xxx - q
        f2 = u_x - z2 + E*I/kapa/A/G * phi_xx
        f1 = self.lagrange_2[:,0:1] * f1
        f2 = self.lagrange_2[:,1:2] * f2
        loss_f1 = tf.reduce_mean(tf.square(f1))
        loss_f2 = tf.reduce_mean(tf.square(f2))
        loss_f = loss_f1 + loss_f2
        return loss_f
    
    def loss(self,x,lb,ub):
        loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss_BC(lb, ub)
        loss_pde = self.loss_PDE(x)        
        return loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables) # trainable_variables is W (w+b)
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
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
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            tf.print(counter, loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4)
        counter += 1
    
    
    def adaptive_gradients(self):
        
        with tf.GradientTape() as tape:
            tape.watch(self.W)
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            loss_val = loss_pde + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4    
        grads = tape.gradient(loss_val,self.W)
        del tape

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(self.lagrange_1)
            tape.watch(self.lagrange_2)
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            loss_val = loss_pde + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
        grads_L1 = tape.gradient(loss_val,self.lagrange_1) # boundary terms
        grads_L2 = tape.gradient(loss_val,self.lagrange_2) # residual terms
        del tape
        
        return loss_val, grads, grads_L1, grads_L2
        
##############################################################################
# DATA PREP #
train = np.loadtxt("train.dat")*L
test = np.loadtxt("test.dat")*L
x_train = train[:,0:1]
x_test = test[:,0:1]

counter = 0

lb = np.array([np.min(x_train)])[:,None]
ub = np.array([np.max(x_train)])[:,None]

##############################################################################
# MODEL TRAINING AND TESTING #

layers = np.array([1,50,50,50,50,2]) 
PINN = Sequentialmodel(layers)
init_params = PINN.get_weights().numpy() 
start_time = time.time()

###################
# Self adaptive
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer_L1 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer_L2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

num_epochs = 5000

for epoch in range(num_epochs):
    
        loss_value, grads, grads_L1, grads_L2 = PINN.adaptive_gradients()

        if epoch % 100 == 0:
            tf.print(epoch, loss_value)
        
        optimizer.apply_gradients(zip(grads, PINN.W)) #gradient descent weights 
        optimizer_L1.apply_gradients(zip([-grads_L1], [PINN.lagrange_1])) # gradient ascent adaptive coefficients of boundary residual
        optimizer_L2.apply_gradients(zip([-grads_L2], [PINN.lagrange_2])) # gradient ascent adaptive coefficients of PDE residual
              
init_params = PINN.get_weights().numpy()


###################

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
                                          'maxfun': 50000, 
                                          'maxiter': 10000, 
                                          'iprint': -1,
                                          'maxls': 50})

elapsed = time.time() - start_time
print('Training time: %.2f' %(elapsed))

x_train = x_train[x_train[:, 0].argsort()]
x_test = x_test[x_test[:, 0].argsort()]


y_train = PINN.evaluate(x_train)
y_test = PINN.evaluate(x_test)

fig, ax = plt.subplots()

ax.plot(x_train, y_train[:,0:1],'bo', linewidth=2.0)
ax.plot(x_test, y_test[:,0:1], linewidth=2.0)
ax.set(xlim=(-1, 101), xticks=np.arange(-1, 101),
       ylim=(-1.1, 0.1), yticks=np.arange(-1, 0.1))

plt.show()

y_max = PINN.evaluate([[L/2]])
y_max = y_max[0,0].numpy()
y_e  = q*L**4/384/E/I
tf.print(y_max, y_e)







