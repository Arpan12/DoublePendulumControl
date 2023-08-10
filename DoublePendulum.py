from scipy.integrate import odeint 
import sys
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from casadi import *


m1 = 1
l1 = 1
m2 = 1
l2 = 1
I1 = m1*(l1**2)
I2 = m2*(l2**2)


def deriv2(y,u):
    theta1 = y[0]
    theta2 = y[1]
    y1_dot = y[2]
    y2_dot = y[3]
    g = 9.81
    s1 = np.sin(theta1)


    M = np.array([[I1+I2+m2*l1**2+2*m2*l1*l2*np.cos(theta2),I2+m2*l1*l2*np.cos(theta2)],
         [I2+m2*l1*l2*np.cos(theta2),I2]])
    print(M.shape)
    C = np.array([[-2*m2*l1*l2*np.sin(theta2)*y2_dot, -m2*l1*l2*np.sin(theta2)*y2_dot],
         [m2*l1*l2*np.sin(theta2)*y1_dot,0]])
    print(C.shape)
    T = np.array([[-m1*g*l1*s1-m2*g*(l1*s1+l2*np.sin(theta1+theta2))],[-m2*g*l2*np.sin(theta1+theta2)]])
    print(T.shape)
    B = np.array([[0],[1]])
    print(B.shape)
    d2y = np.matmul(np.linalg.inv(M),T+np.matmul(B,u)-np.matmul(C,[[y1_dot],[y2_dot]]))
    return np.array([y1_dot,y2_dot,d2y[0][0],d2y[1][0]])


def deriv(y,t,u,l):
    theta1 = y[0]
    theta2 = y[1]
    y1_dot = y[2]
    y2_dot = y[3]
    g = 9.81
    s1 = MX.sin(theta1)

    M = MX(2,2)
    M[0,0] = I1+I2+m2*l1**2+2*m2*l1*l2*MX.cos(theta2)
    M[0,1] = I2+m2*l1*l2*MX.cos(theta2)
    M[1,0] = I2+m2*l1*l2*MX.cos(theta2)
    M[1,1] = I2
    
    C = MX(2,2)
    C[0,0] = -2*m2*l1*l2*MX.sin(theta2)*y2_dot
    C[0,1] = -m2*l1*l2*MX.sin(theta2)*y2_dot
    C[1,0] = m2*l1*l2*MX.sin(theta2)*y1_dot
    C[1,1] = 0

    T = MX(2,1)
    T[0,0] = -m1*g*l1*s1-m2*g*(l1*s1+l2*MX.sin(theta1+theta2))
    T[1,0] = -m2*g*l2*MX.sin(theta1+theta2)
   
    B = MX(2,1)
    B[0,0] = 0
    B[1,0] = 1
    
    y_dot = MX(2,1)
    y_dot[0,0]=y1_dot 
    y_dot[1,0] = y2_dot

    d2y = solve(M,T+mtimes(B,u)-mtimes(C,y_dot))
    return y1_dot,y2_dot,d2y[0][0],d2y[1][0]

fig = plt.figure(figsize=(8.33,6.25),dpi=72)
ax = fig.add_subplot(111)
y = [math.pi/3,math.pi/2,0,0]
x1 = math.sin(y[0])
y1 = - math.cos(y[0])
x2 = x1+math.sin(y[1])
y2 = y1-math.cos(y[1])
line, = ax.plot([0,x1,x2],[0,y1,y2],lw=2,c='k')
r=0.05
c1 = Circle((x1,y1),r,fc='b',ec='b',zorder=10)
c2 = Circle((x2,y2),r,fc='b',ec='b',zorder=10)
circle1 = ax.add_patch(c1)
circle2  = ax.add_patch(c2)
ns=20
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)



def animate(i):
    
    x1 = math.sin(y_track[i,0])
    y1 = - math.cos(y_track[i,0])
    x2 = x1+math.sin(y_track[i,1])
    y2 = y1-math.cos(y_track[i,1])
    line.set_data([0,x1,x2],[0,y1,y2])
    circle1.set_center((x1,y1))
    circle2.set_center((x2,y2))
    


####controls code
N=100
dt=0.1
T =10

theta1 = MX.sym('theta1')
theta2 = MX.sym('theta2')
dtheta1 = MX.sym('dtheta1')
dtheta2 = MX.sym('dtheta2')
x = vertcat(theta1,theta2,dtheta1,dtheta2)

u = MX.sym('u')


y = deriv(x,dt,u,1)
xdot = vertcat(y[0],y[1],y[2],y[3])

#s = theta1-(int)(theta1/(2*np.pi))*(2*np.pi)
L =  0.5*(u)**2 + 100*(MX.cos(theta1)+1)**2+100*(MX.cos(theta2)+1)**2+10*dtheta1**2+10*dtheta2**2


M = 4 # RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [xdot, L])
X0 = MX.sym('X0', 4)
U = MX.sym('U')
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + DT/2 * k1, U)
    k3, k3_q = f(X + DT/2 * k2, U)
    k4, k4_q = f(X + DT * k3, U)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [X0, U], [X, Q],['x0','u'],['xf','qf'])

Fk = F(x0=[0.0,0.0,0.0,0.0],u=0.0)
print(Fk['xf'])
print(Fk['qf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 4)
w += [Xk]
lbw += [0.1, 0.0,0,0]
ubw += [0.1, 0.0,0,0]
w0 += [0.1, 0.0,0,0]

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w   += [Uk]
    lbw += [-inf]
    ubw += [inf]
    w0  += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, u=Uk)
    Xk_end = Fk['xf']
    J=J+Fk['qf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 4)
    w   += [Xk]
    lbw += [-inf, -inf,-inf,-inf]
    ubw += [ inf,inf,inf,inf]
    w0  += [0,0,0,0]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0, 0,0,0]
    ubg += [0, 0,0,0]

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)


sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Plot the solution
x1_opt = w_opt[0::5]
x2_opt = w_opt[1::5]
x3_opt = w_opt[2::5]
x4_opt = w_opt[3::5]
u_opt = w_opt[4::5]

print("x1 = ")
print(x1_opt)
print("x2 = ")
print(x2_opt)
print("u = ")
print(u_opt)


####
y_init = [0.1,0,0,0]
X = y_init
t = [0,0.1]

y_track=np.array([y_init for i in range(100)])
print(y_track.shape)
for i in range(0,99):
    y_track[i,:] = [x1_opt[i],x2_opt[i],x3_opt[i],x4_opt[i]]
    

ani = animation.FuncAnimation(fig, animate,frames=99,interval=100)
#plt.show()
FFwriter = animation.FFMpegWriter(fps=10)
ani.save('animation.mp4', writer = FFwriter)
