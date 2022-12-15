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
import seaborn as sns
import codecs, json

# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

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
    idx = np.randome.choice(all_X_u_train.shape[0], N_u, replace = False)
    '''np,random.choice(n,size, replace=False) 
    Generate a uniform random sample from np.range(n)of size "size" without replacement'''
    X_u_train = all_X_u_train[idx,:] # choose indices from set 'idx' (x,t)
    u_train = all_u_train[idx,:] # choose corresponding u
    
    '''Collocation Points'''
    # LAtin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub-lb) * lhs(2,N_f) 
    '''lhs(2,N_f): produces matrix (2,5) with value between zero and one (Latin Hypercube Sampling)'''
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points
    
    return X_f_train, X_u_train, u_train

# Check trainingdata again before doing new one