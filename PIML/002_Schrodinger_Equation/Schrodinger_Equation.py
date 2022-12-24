import tensorflow as tf
import datetime, os
#hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs
import seaborn as sns
import codecs, json
import pickle

''' 002_Schrodinger.mat
ih_t + 0.5h_xx + |h|**2 h =0;   IC: h(0,x) = 2sech(x)
                                BC: h(t,-5)= h(t,5); h_x(t,-5)=h_x(t,5)'''

np.random.seed(1234)
tf.random.set_seed(1234)

print("TensorFlow version: {}".format(tf.__version__))


##############################################################################
# DATA PREP #
# Training and Testing data is prepared from the solution file
data = scipy.io.loadmat('../000_Data/002_Schrodinger.mat') #Load data from file
x = data['x'] #256 points between -5 and 5 (256,1)
t = data['t'] #201 time points between 0 and pi/2 (201,1)
sol = data['usol'] #solution of (256,201) grid points
usol = np.real(sol) # u (256,201)
# np.real: Return the real part of the complex argument.
vsol = np.imag(sol) # u (256,201)
# np.imag: Return the imaginary part of the complex argument.
hsol = np.sqrt(usol**2 + vsol**2) # h (256,201)

#makes 2 array X and T such that u(X[i],T[j])=usol[i][j] are a tuple
X, T = np.meshgrid(x,t) # X(201,256), T(201,256)


##############################################################################
# TEST DATA #
# We prepare the test data to compare against the solution produced by the PINN.
# X_test = [X[i],T[i]] (51456, 2) for interpolation 
X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
'''X.flatten(): [r1
                r2
                .
                .
                rn] --> X[51456,0]; X[51456,0][:,None] --> X[51456,1]
hstack: Stacks array in sequence horizonally (column wise)'''

# Domain bounds
lb = X_test[0] # [-5. 0.] # First row (2,0)
ub = X_test[-1] # [5 pi/2] # Last row (2,0)
'''Fortran Style ('F') flatten, stacked column wise!
    u = [c1
         c2
         .
         .
         cn]
    u_test(51456,1)'''
u_test = usol.flatten('F')[:,None]
v_test = vsol.flatten('F')[:,None]
h_test = hsol.flatten('F')[:,None]


##############################################################################
# TRAINING DATA #
# The boundary conditions serve as the test data for the PINN and the collocation
# points are generated using Latin Hypercube Sampling
def trainingdata(N0, N_b, N_f):
    '''Boundary conditions'''
    # Ininital Condition -5 <= x <= 5 and t = 0 --> Find X0
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1(256,2)
    leftedge_u = usol[:,0][:,None] #(256,1)
    leftedge_v = vsol[:,0][:,None] #(256,1)
    
    # Boundary Condition x = -5 and 0 <= t < pi/2 --> Find Xlb
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2(201,2)
    
    # Boundary Condition x = 5 and 0 <= t < pi/2 --> Find Xub
    topedge_x = np.hstack((X[:,-1][:,None],T[:,0][:,None])) #L3(201,2)
    
    # Find X0, u0. v0
    idx0 = np.random.choice(leftedge_x.shape[0], N0, replace = False)
    X0 = leftedge_x[idx0,:] # choose indices from set 'idx0' (x,t)
    u0 = leftedge_u[idx0,:] # choose corresponding u0
    v0 = leftedge_v[idx0,:] # choose corresponding v0
        
    # Find Xlb, Xub
    idx_b = np.random.choice(bottomedge_x.shape[0], N_b, replace = False)
    Xlb = bottomedge_x[idx_b,:] # choose indices from set 'idx_b' (x,t)
    Xub = topedge_x[idx_b,:] # choose indices from set 'idx_b' (x,t)
    
    '''Collocation Points'''
    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f = lb + (ub-lb) * lhs(2,N_f) 
    # X_f_train = np.vstack((X_f_train, X_train)) # append training points to collocation points
    '''Check whether we need to append or not?'''
    
    return X0, u0, v0, Xlb, Xub, X_f


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
            
            #weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            
            w = tf.Variable(w,trainable=True,name='w'+str(i+1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]),dtype='float64'),trainable=True,name='b'+str(i+1))
            self.W.append(w)
            self.W.append(b)
            
            self.parameters += input_dim * output_dim + output_dim
            
    def evaluate(self,x):
        x = 2*(x-lb)/(ub-lb)-1 # old: (x-lb)/(ub-lb)
        a = x
        
        for i in range(len(layers)-2):
            z = tf.add(tf.matmul(a,self.W[2*i]),self.W[2*i+1]) 
            a = tf.nn.tanh(z)
        
        a = tf.add(tf.matmul(a,self.W[-2]),self.W[-1]) # For regression, no activation to last layer
        a_u = a[:,0][:,None]
        a_v = a[:,1][:,None]
        return a_u, a_v
    
    def get_weights(self):
        parameters_1d = [] # [.... W_i, b_i ....] 1D array
        
        for i in range(len(layers)-1):
            w_1d = tf.reshape(self.W[2*i],[-1]) # Flatten weights (..,0)
            b_1d = tf.reshape(self.W[2*i+1],[-1]) # Flatten biases
            
            parameters_1d = tf.concat([parameters_1d,w_1d],0) # concat weights
            parameters_1d = tf.concat([parameters_1d,b_1d],0) # concat biases
        return parameters_1d
    
    def set_weights(self, parameters):
        for i in range(len(layers)-1):
            shape_w = tf.shape(self.W[2*i]).numpy() # Shape of the weight tensor
            size_w = tf.size(self.W[2*i]).numpy() # Size of the weight tensor
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # Shape of the bias tensor
            size_b  = tf.size(self.W[2*i+1]).numpy() # Size of the bias tensor
            
            pick_w = parameters[0:size_w] # Pick the weights
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) 
            parameters = np.delete(parameters, np.arange(size_w),0) # delete
            
            pick_b = parameters[0:size_b] # Pick the biases
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign
            parameters = np.delete(parameters, np.arange(size_b),0) # delete
            
    def loss_BC(self,x0, u0, v0, xlb, xub):
        u_pred, v_pred = self.evaluate(x0)
        loss_u0 = tf.reduce_mean(tf.square(u0-u_pred))
        loss_v0 = tf.reduce_mean(tf.square(v0-v_pred))
        
        g_lb = tf.Variable(xlb, dtype='float64', trainable=False)
        g_ub = tf.Variable(xub, dtype='float64', trainable=False)
        x_lb = g_lb[:,0:1]
        x_ub = g_ub[:,0:1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_lb)
            tape.watch(x_ub)
            u_lb, v_lb = self.evaluate(x_lb)
            u_ub, v_ub = self.evaluate(x_ub)
            
        u_x_lb = tape.gradient(u_lb,x_lb)
        v_x_lb = tape.gradient(v_lb,x_lb)
            
        u_x_ub = tape.gradient(u_ub,x_ub)
        v_x_ub = tape.gradient(v_ub,x_ub)
        del tape
        loss_u_b = tf.reduce_mean(tf.square(u_lb - u_ub)) + tf.reduce_mean(tf.square(u_x_lb - u_x_ub)) 
        loss_v_b = tf.reduce_mean(tf.square(v_lb - v_ub)) + tf.reduce_mean(tf.square(v_x_lb - v_x_ub))
        
        loss_u = loss_u0 + loss_u_b
        loss_v = loss_v0 + loss_v_b
        # tf.reduce_mean(x): computes the mean of elements across dimensions of a tensor.
        return loss_u, loss_v
    
    def loss_PDE(self, x_f):
        g = tf.Variable(x_f, dtype='float64', trainable=False)        
        x_f = g[:,0:1]
        t_f = g[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)
            # Trainable variables are automatically watched.
            g = tf.stack([x_f[:,0],t_f[:,0]], axis=1)
            u, v = self.evaluate(g)
            u_x = tape.gradient(u,x_f)
            u_xx = tape.gradient(u_x,x_f)
            v_x = tape.gradient(v,x_f)
            v_xx = tape.gradient(v_x,x_f)

        u_t = tape.gradient(u,t_f)
        v_t = tape.gradient(v,t_f)
        del tape
        
        # Schrodinger = ih_t +0.5h_xx +|h|^2 h, IC: 2sech(x)
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        loss_f_u = tf.reduce_mean(tf.square(f_u))
        loss_f_v = tf.reduce_mean(tf.square(f_v))
        return loss_f_u, loss_f_v
    
    def loss(self, x0, u0, v0, xlb, xub, g):
        loss_u, loss_v = self.loss_BC(x0, u0, v0, xlb, xub)
        loss_f_u, loss_f_v = self.loss_PDE(g)
        loss = loss_u + loss_v + loss_f_u + loss_f_v
        
        return loss, loss_u, loss_v, loss_f_u, loss_f_v
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables) # trainable_variables is W (w+b)
            loss_val, loss_u, loss_v, loss_f_u, loss_f_v = self.loss(X0, u0, v0, Xlb, Xub, X_f)
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
        loss_value, loss_u, loss_v, loss_f_u, loss_f_v = self.loss(X0, u0, v0, Xlb, Xub, X_f)
        u_pred, v_pred = self.evaluate(X_test)
        h_pred = np.sqrt(u_pred**2 + v_pred**2)
        error_u = np.linalg.norm((u_test-u_pred),2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm((v_test-v_pred),2)/np.linalg.norm(v_test,2)
        error_h = np.linalg.norm((h_test-h_pred),2)/np.linalg.norm(h_test,2)
        
        tf.print(loss_value, loss_u, loss_v, loss_f_u, loss_f_v)
        print('Error u: %.5f, v:%.5f, h:%.5f' % (error_u, error_v, error_h))

##############################################################################
# SOLUTION PLOT #
def solutionplot(h_pred, X_train, h_train):
    fig, ax = plt.subplots() # Create a figure and a set of subplots.
    ax.axis('off') # Turns off axes in subplots.
    
    gs0 = gridspec.GridSpec(1,2) # A grid layout to place subplots within a figure.
    gs0.update(top=1-0.06,bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax=plt.subplot(gs0[:,:])
    
    h = ax.imshow(h_pred, interpolation='nearest', cmap='rainbow',
                  extent=[T.min(), T.max(), X.min(), X.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_train[:,1],X_train[:,0], 'kx', label = 'Data (%d points)' 
            %(h_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[100]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[150]*np.ones((2,1)), line, 'w-', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$h(x,t)$', fontsize=10)
    
    
    # Row 1: u(t,x) slices
    gs1 = gridspec.GridSpec(1,3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0,0])
    ax.plot(x, hsol.T[50,:], 'b-', linewidth=2, label='Exact')
    ax.plot(x, h_pred.T[50,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$h(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-2.1,2.1])
    
    ax = plt.subplot(gs1[0,1])
    ax.plot(x,hsol.T[100,:],'b-', linewidth=2, label='Exact')
    ax.plot(x, h_pred.T[100,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$h(x,t)$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_title('$t = 0.5s$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0,2])
    ax.plot(x,hsol.T[150,:], 'b-', linewidth=2, label='Exact')
    ax.plot(x,h_pred.T[150,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$h(x,t)$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_title('$t = 0.75s$', fontsize=10)
    
    plt.savefig('Schrodinger.png', dpi=500)
    

##############################################################################
# MODEL TRAINING AND TESTING #
'''
A function "Model" is defined to generate a NN as per the input set of hyperparameters,
which is then trained and tested. The L2 Norm of the solution error is returned as 
a comparison metric'''
N0 = 50
N_b = 50 # Total number of data points for 'u'
N_f = 20000 # Total number of collocation points

# Training data
X0, u0, v0, Xlb, Xub, X_f = trainingdata(N0, N_b, N_f)

layers = np.array([2,100,100,100,100,2]) # 4 hidden layers

PINN = Sequentialmodel(layers)

init_params = PINN.get_weights().numpy() # [.... W_i, b_i ....] 1D array

start_time = time.time()
#This method returns the time as a floating point number expressed in seconds since the epoch, in UTC.

# train the model with Scipy L-BFGS optimizer
results = scipy.optimize.minimize(fun=PINN.optimizerfunc,
                                  x0 = init_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac=True, # If jac is True, fun is assumed to return
                                            #the gradient along with the objective function
                                  callback=PINN.optimizer_callback,
                                  options={'disp':None,
                                          'maxcor': 50,
                                          'ftol': 1*np.finfo(float).eps,
                                          'gtol': 5e-2, # 5e-8
                                          'maxfun': 5, # 50000
                                          'maxiter': 5, # 50000
                                          'iprint': -1,
                                          'maxls': 50})

with open("Results.pkl","wb") as f:
    pickle.dump(results,f)

del f

elapsed = time.time() - start_time
print('Training time: %.2f' %(elapsed))

print(results)

PINN.set_weights(results.x)

'''Model Accuracy'''
u_pred, v_pred = PINN.evaluate(X_test)
h_pred = np.sqrt(u_pred**2 + v_pred**2)

error_u = np.linalg.norm((u_test-u_pred),2)/np.linalg.norm(u_test,2)
error_v = np.linalg.norm((v_test-v_pred),2)/np.linalg.norm(v_test,2)
error_h = np.linalg.norm((h_test-h_pred),2)/np.linalg.norm(h_test,2)
print('Error u: %.5f' % (error_u))
print('Error v: %.5f' % (error_v))
print('Error h: %.5f' % (error_h))

h_pred = np.sqrt(u_pred**2 +v_pred**2)
h0 = np.sqrt(u0**2 +v0**2)

h_pred = np.reshape(u_pred, (256,201), order='F')


'''Solution Plot'''
solutionplot(h_pred, X0, h0)


##############################################################################
# PLOT OF COLLOCATION POINTS #
N0 = 50
N_u = 50 # Total number of data points for 'u'
N_f = 20000 # Total number of collocation points

# Training data
X0, u0, v0, Xlb, Xub, X_f = trainingdata(N0, N_u, N_f)

fig, ax = plt.subplots()

plt.plot(X0[:,1], X0[:,0], '*', color = 'red', markersize = 5, 
         label = 'Boundary collocation = 100')
plt.plot(X_f[:,1], X_f[:,0], 'o', markersize = 0.5, 
         label = 'PDE collocation = 10,000')

plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Collocation points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

fig.savefig('collocation_points_Schrodinger.png', dpi = 500, bbox_inches='tight')











