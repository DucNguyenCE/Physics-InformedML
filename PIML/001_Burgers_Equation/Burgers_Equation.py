import tensorflow as tf
import datetime, os
#hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}
#0 (default) show all, 1 to filter out INFOR logs, 2 to additionally filter out
#WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs #Latin Hypercube Sampling
'''Latin Hypercube sampling is a method that can be used to sample random
numbers in which samples are distributed evenly over the same space.
It is widely used to generate samples that are known as controlled random
samples and it often applied in Monte Carlo analysis because it can dramatically
reduce the number of simulation needed to achive accurate results.'''
import seaborn as sns
import codecs, json

''' 001_burgers_shock_mu_01_pi.mat
u_t + u*u_x - (0.01/pi)*u_xx = 0, -1<=x <=1, 0<= t < 1
                         IC: -sin(pi*x), 
                         BC: u(−1, t) = u(1, t) = 0 '''

np.random.seed(1234)
# generates same random numbers each time
# pseudo-random numbers: a number that's almost random, but not really random.
#                      : a computer-generated random number.
#                      : are computer generated numbers that appear random, but actually predetermined.
'''If you ask the same question, you will get the same answer every time.
Like: If the input is the same, then output should be the same.''' 
# This help us that when we run the code again, we get the same result.
# np.random.seed makes your code repeatable and easier to share.
tf.random.set_seed(1234)
# Same as numpy but set for tensorflow


print("TensorFlow version: {}".format(tf.__version__))


##############################################################################
# DATA PREP #
# Training and Testing data is prepared from the solution file
data = scipy.io.loadmat('../000_Data/001_burgers_shock_mu_01_pi.mat') #Load data from file
x = data['x'] #256 points between -1 and 1 (256,1)
t = data['t'] #100 time points between 0 and 1 (100,1)
usol = data['usol'] #solution of (256,100) grid points
#makes 2 array X and T such that u(X[i],T[j])=usol[i][j] are a tuple
X, T = np.meshgrid(x,t) # X(100,256), T(100,256)


##############################################################################
# TEST DATA #
# We prepare the test data to compare against the solution produced by the PINN.
# X_u_test = [X[i],T[i]] (25600, 2) for interpolation 
X_u_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
'''X.flatten(): [r1
                r2
                .
                .
                rn] --> X[25600,0]; X[25600,0][:,None] --> X[25600,1]
hstack: Stacks array in sequence horizonally (column wise)'''

# Domain bounds
lb = X_u_test[0] # [-1. 0.] # First row (2,0)
ub = X_u_test[-1] # [1 0.99] # Last row (2,0)

'''
    Fortran Style ('F') flatten, stacked column wise!
    u = [c1
         c2
         .
         .
         cn]
    u(25600,1)
'''
u = usol.flatten('F')[:,None]


##############################################################################
# TRAINING DATA #
# The boundary conditions serve as the test data for the PINN and the collocation
# points are generated using Latin Hypercube Sampling
def trainingdata(N_u, N_f):
    '''Boundary conditions'''
    # Ininital Condition -1 <= x <= 1 and t = 0
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1(256,2)
    leftedge_u = usol[:,0][:,None] #(256,1)
    
    # Boundary Condition x = -1 and 0 <= t < 1
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2(100,2)
    bottomedge_u = usol[-1,:][:,None] #(100,1)
    
    # Boundary Condition x = 1 and 0 <= t < 1
    topedge_x = np.hstack((X[:,-1][:,None],T[:,0][:,None])) #L3(100,2)
    topedge_u = usol[0,:][:,None] #(100,1)
    
    # X_u_train [456,2] (456 = 256[L1] + 100[L2] +100[L3])
    all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x])
    '''vstack: Stacks array in sequence vertically (row wise)'''
    
    # corresponding u [456,1]
    all_u_train = np.vstack([leftedge_u,bottomedge_u,topedge_u])
    
    # Choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace = False)
    '''np.random.choice(n,size, replace=False) 
    Generate a uniform random sample from np.range(n)of size "size" without replacement
    replacement: default is true, meaning that a value of a can be selected mutiple times.'''
    X_u_train = all_X_u_train[idx,:] # choose indices from set 'idx' (x,t)
    u_train = all_u_train[idx,:] # choose corresponding u
    
    '''Collocation Points'''
    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub-lb) * lhs(2,N_f) 
    '''lhs(2,N_f): produces matrix (2,N_f) with value between zero and one (Latin Hypercube Sampling)
    lhs: it partitions each input distribution into N intervals of equal probability, and selects one sameple 
    from each interval.'''
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points
    
    return X_f_train, X_u_train, u_train


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
            
            # It's important not to skip initialization.
            # Chosing the right initializer is important to our model's performance and training.
            # Uniform Xavier initianlization: draw each weight, w, from a random uniform
            # distribution in [-x,x] for x = sqrt(6.0/(inputs+outputs))
            # Normal Xavier initialization: draw each weight, w, from a normal distribution wich
            # mean of 0, and a standard deviation = sqrt(2.0/(inputs+outputs))
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))
            # Xavier initialization: also known as Glorot initialization.
            
            #weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            
            w = tf.Variable(w,trainable=True,name='w'+str(i+1))
            # The distinction between trainable True and False is used to let 
            # Optimizers know which variables they can act upon
            # trainable=True: automatically adds the variable to the 
            # GraphKeys.TRAINABLE_VARIABLES collection
            b = tf.Variable(tf.cast(tf.zeros([output_dim]),dtype='float64'),trainable=True,name='b'+str(i+1))
            # tf.zeros([n]) create a zeros vector (n,0)
            # tf.cast(): casts a tensor to a new type.
            self.W.append(w)
            # .append(w): Append values to the end of an array.
            self.W.append(b)
            
            self.parameters += input_dim * output_dim + output_dim
            
    def evaluate(self,x):
        x = (x-lb)/(ub-lb)
        a = x
        
        for i in range(len(layers)-2):
            z = tf.add(tf.matmul(a,self.W[2*i]),self.W[2*i+1]) 
            # tf.matlul: multiplies matrix a by matrix b; a*x
            # tf.add: returns the addition of the two tf.tensor objects element wise. a*x + b
            a = tf.nn.tanh(z)
            # tf.nn.tanh(z): computes hyperbolic tangent of z element-wise.
        
        a = tf.add(tf.matmul(a,self.W[-2]),self.W[-1]) # For regression, no activation to last layer
        return a
    
    def get_weights(self):
        parameters_1d = [] # [.... W_i, b_i ....] 1D array
        
        for i in range(len(layers)-1):
            w_1d = tf.reshape(self.W[2*i],[-1]) # Flatten weights (..,0)
            # tf.reshape(tensor, shape): a shape of [-1] flattens into 1-D.
            b_1d = tf.reshape(self.W[2*i+1],[-1]) # Flatten biases
            
            parameters_1d = tf.concat([parameters_1d,w_1d],0) # concat weights
            # tf.concat(values, axis): concatenates tensors along one dimension.
            # axis=0: row, axis=1: column
            parameters_1d = tf.concat([parameters_1d,b_1d],0) # concat biases
        return parameters_1d
    
    def set_weights(self, parameters):
        for i in range(len(layers)-1):
            shape_w = tf.shape(self.W[2*i]).numpy() # Shape of the weight tensor
            # .numpy(): converts tf.Variable to a numpy array.
            size_w = tf.size(self.W[2*i]).numpy() # Size of the weight tensor
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # Shape of the bias tensor
            size_b  = tf.size(self.W[2*i+1]).numpy() # Size of the bias tensor
            
            pick_w = parameters[0:size_w] # Pick the weights
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) 
            # assign(): assign a new value to a tf.Variable().
            # Assign 1-D to 2-D so it has to be reshape before.
            parameters = np.delete(parameters, np.arange(size_w),0) # delete
            # np.delete: Return a new array with sub-arrays along an axis deleted. 
            # For a one dimensional array, this returns those entries not returned by arr[obj].
            
            pick_b = parameters[0:size_b] # Pick the biases
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign
            parameters = np.delete(parameters, np.arange(size_b),0) # delete
            
    def loss_BC(self,x,y):
        loss_u = tf.reduce_mean(tf.square(y-self.evaluate(x)))
        # tf.reduce_mean(x): computes the mean of elements across dimensions of a tensor.
        return loss_u
    
    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        nu = 0.01/np.pi
        
        x_f = g[:,0:1]
        t_f = g[:,1:2]
        # tf.GradientTape: Record operations for automatic differentiation.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)
            # Trainable variables are automatically watched.
            g = tf.stack([x_f[:,0],t_f[:,0]], axis=1)
            
            z = self.evaluate(g)
            u_x = tape.gradient(z,x_f) 
        # Computes the gradient using operations recorded in context of this tape.
        u_t = tape.gradient(z,t_f)
        u_xx = tape.gradient(u_x, x_f)
        del tape
        
        # Burgers equation:u + u*u_x - nu*u_xx = 0, IC: -sin(pi*x)
        f = u_t + (self.evaluate(g))*(u_x) - (nu) * u_xx 
        loss_f = tf.reduce_mean(tf.square(f))
        return loss_f
    
    def loss(self,x,y,g):
        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(g)
        loss = loss_u + loss_f
        
        return loss, loss_u, loss_f
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables) # trainable_variables is W (w+b)
            loss_val, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)
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
        loss_value, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)
        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)
        # np.linalg.norm(x,2): This is the square root of the sum of squared elements 
        # and can be interpreted as the length of the vector in Euclidean space.
        # Note: np.linalg.norm(x,2) = sqrt(|x1|**2+|x2|**2|)
        # Note: np.linalg.norm(x,3) = (|x1|**3 + |x2|**3)**(1/3)
        tf.print(loss_value, loss_u, loss_f, error_vec)
    

##############################################################################
# SOLUTION PLOT #
def solutionplot(u_pred, X_u_train, u_train):
    fig, ax = plt.subplots() # Create a figure and a set of subplots.
    ax.axis('off') # Turns off axes in subplots.
    
    gs0 = gridspec.GridSpec(1,2) # A grid layout to place subplots within a figure.
    gs0.update(top=1-0.06,bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax=plt.subplot(gs0[:,:])
    
    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow',
                  extent=[T.min(), T.max(), X.min(), X.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1],X_u_train[:,0], 'kx', label = 'Data (%d points)' 
            %(u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$u(x,t)$', fontsize=10)
    '''
    Slices of the solution at points t=0.25, t=0.50 and t=0.75
    '''
    
    # Row 1: u(t,x) slices
    gs1 = gridspec.GridSpec(1,3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0,0])
    ax.plot(x, usol.T[25,:], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[25,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0,1])
    ax.plot(x,usol.T[50,:],'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[50,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.5s$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0,2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[75,:], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75s$', fontsize=10)
    
    plt.savefig('Burgers.png', dpi=500)
    

##############################################################################
# MODEL TRAINING AND TESTING #
'''
A function "Model" is defined to generate a NN as per the input set of hyperparameters,
which is then trained and tested. The L2 Norm of the solution error is returned as 
a comparison metric'''
N_u = 100 # Total number of data points for 'u'
N_f = 10000 # Total number of collocation points

# Training data
X_f_train, X_u_train, u_train = trainingdata(N_u, N_f)

layers = np.array([2,20,20,20,20,20,20,20,20,1]) # 8 hidden layers

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
                                          'maxcor': 200,
                                          'ftol': 1*np.finfo(float).eps,
          # The iteration stops when (f^k-f^{k+1})/max{|f^k|,|f^{k+1}|,1} <=ftol
                                          'gtol': 5e-8,
          # The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol 
          # where pg_i is the i-th component of the projected gradient.
                                          'maxfun': 50000, # Maximum number of function evaluations.
                                          'maxiter': 5000, # Maximum number of iterations.
                                          'iprint': -1,
          # Controls the frequency of output. iprint < 0 means no output.
                                          'maxls': 50})
          # Maximum number of line search steps (per iteration). Default is 20.
'''Limited-memory BFGS (L-BFGS): an optimization algorithm in the family of 
quasi-Newton methods that approximates the Broyden-Fletcher-Goldfarb-Shanno algorithm
using a limited amount of computer memory.
The algorithm's target problem is to minimize f(x) over unconstrained values of 
the real-vector x where f is a differentiable scala function.

The L-BFGS-B algorithm extends L-BFGS to handle simple box constraints (aka bound 
constraints) on variables; that is, constraints of the form li ≤ xi ≤ ui where li 
and ui are per-variable constant lower and upper bounds, respectively (for each xi, 
either or both bounds may be omitted). The method works by identifying fixed and 
free variables at every step (using a simple gradient method), and then using the 
L-BFGS method on the free variables only to get higher accuracy, and then repeating the process.
'''

elapsed = time.time() - start_time
print('Training time: %.2f' %(elapsed))

print(results)

PINN.set_weights(results.x)

'''Model Accuracy'''
u_pred = PINN.evaluate(X_u_test)
error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)
print('Test Error: %.5f' % (error_vec))

u_pred = np.reshape(u_pred, (256,100), order='F')

'''Solution Plot'''
solutionplot(u_pred, X_u_train, u_train)


##############################################################################
# PLOT OF COLLOCATION POINTS #
N_u = 100 # Total number of data points for 'u'
N_f = 10000 # Total number of collocation points

# Training data
X_f_train, X_u_train, u_train  = trainingdata(N_u, N_f)

fig, ax = plt.subplots()

plt.plot(X_u_train[:,1], X_u_train[:,0], '*', color = 'red', markersize = 5, 
         label = 'Boundary collocation = 100')
plt.plot(X_f_train[:,1], X_f_train[:,0], 'o', markersize = 0.5, 
         label = 'PDE collocation = 10,000')

plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Collocation points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

fig.savefig('collocation_points_Burgers.png', dpi = 500, bbox_inches='tight')











