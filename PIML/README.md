# Information

I. DATA-DRIVEN SOLUTION OF PARTIAL DIFFERENTIAL EQUATIONS
I.1. Continuous time models
I.1.1. Burger's equation
In one space dimension, the Burger’s equation along with Dirichlet boundary conditions reads as
u_t + uu_x - (0.01/pi)u_xx = 0, x in [-1,1], t in [0,1]
u(0,x) = -sin(pi*x)
u(t,-1) = u(t,1) = 0

I.1.2. Schrodinger equation
The nonlinear Schrödinger equation along with periodic boundary conditions is given by
ih_t + 0.5h_xx + |h|**2 * h = 0, x in [-5,5], t in [0,pi/2]
h(0,x) = 2sech(x)
h(t,-5) = h(t,5)
h_x(t,-5) = h(t,5)
In fact, if u denotes the real part of h and v is the imaginary part, we are placing a multi-out neural network prior on h(t, x) =  [u(t, x)  v(t, x)].

