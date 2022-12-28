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
            
            u_ub_x = tape.gradient(u_ub,x_ub)
            u_ub_xx = tape.gradient(u_ub_x,x_ub)
            u_ub_xxx = tape.gradient(u_ub_xx,x_ub)
        u_lb_x = tape.gradient(u_lb,x_lb)
        del tape
        loss_bc1 = tf.reduce_mean(tf.square(u_lb))
        loss_bc2 = tf.reduce_mean(tf.square(u_lb_x))
        loss_bc3 = tf.reduce_mean(tf.square(u_ub_xx))
        loss_bc4 = tf.reduce_mean(tf.square(u_ub_xxx))
        return loss_bc1, loss_bc2, loss_bc3, loss_bc4
    
    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        
        x_f = g[:,0:1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)            
            z = self.evaluate(x_f)
            u_x = tape.gradient(z,x_f)
            u_xx = tape.gradient(u_x,x_f)
            u_xxx = tape.gradient(u_xx,x_f)
            u_xxxx = tape.gradient(u_xxx,x_f)
        # Computes the gradient using operations recorded in context of this tape.
        del tape
        
        # Burgers equation:u + u*u_x - nu*u_xx = 0, IC: -sin(pi*x)
        f = u_xxxx + 1 
        loss_f = tf.reduce_mean(tf.square(f))
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
        if counter % 1000 == 0:
            loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4 = self.loss(x_train, lb, ub)
            losst_pde, losst_bc1, losst_bc2, losst_bc3, losst_bc4 = self.loss(x_test, lb, ub)
            tf.print(counter, loss_pde, loss_bc1, loss_bc2, loss_bc3, loss_bc4, losst_pde)
        counter += 1
##############################################################################
# DATA PREP #
train = np.loadtxt("train.dat")
test = np.loadtxt("test.dat")
x_train = train[:,0:1]
x_test = test[:,0:1]
counter = 0


lb = np.array([np.min(x_train)])[:,None]
ub = np.array([np.max(x_train)])[:,None]

##############################################################################
# MODEL TRAINING AND TESTING #

layers = np.array([1,20,20,20,1]) 
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
                                          'maxls': 2000})

elapsed = time.time() - start_time
print('Training time: %.2f' %(elapsed))

print(results)

PINN.set_weights(results.x)

x_train = x_train[x_train[:, 0].argsort()]
x_test = x_test[x_test[:, 0].argsort()]


y_train = PINN.evaluate(x_train)
y_test = PINN.evaluate(x_test)

fig, ax = plt.subplots()

ax.plot(x_train, y_train,'bo', linewidth=2.0)
ax.plot(x_test, y_test, linewidth=2.0)
ax.set(xlim=(-0.1, 1.1), xticks=np.arange(0, 1),
       ylim=(-0.2, 0.1), yticks=np.arange(-0.2, 0.1))

plt.show()










