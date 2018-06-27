# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:13:44 2016

@author: Xiao
"""

import numpy as np
from scipy import integrate
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scipy.linalg as la

###Forward Euler Method 

##delta_t 0.0012

delta_x = 0.05
delta_t = 0.0012
x_eval = np.arange(0.0, 1.0 + delta_x, delta_x)
t_eval = np.arange(0.0, 0.06 + delta_t, delta_t)
ratio = delta_t / (delta_x**2.0)
X, T = np.meshgrid(x_eval, t_eval)
U = np.zeros(X.shape)

for i in range(0,len(x_eval)):
    if x_eval[i] < 0.5:
        U[0,i] = 2.0 * x_eval[i]
    else:
        U[0,i] = 2.0 - 2.0 * x_eval[i]

for k in range(0, len(t_eval) - 1):
    for i in range(1,len(x_eval) - 1):
        U[k + 1, i] = U[k, i] + ratio * (U[k, i + 1] - 2.0 * U[k, i] + U[k, i - 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(T, X, U, rstride = 10, cstride=1)
plt.title('Forward Euler Method with Delta t 0.0012')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0, 0.06])
plt.ylim([0, 1.0])

##delta_t 0.0013

delta_t = 0.0013
t_eval = np.arange(0.0, 0.06+delta_t, delta_t)
ratio = delta_t / (delta_x**2.0)
X, T = np.meshgrid(x_eval, t_eval)
U = np.zeros(X.shape)

for i in range(0, len(x_eval)):
    if x_eval[i] < 0.5:
        U[0,i] = 2.0 * x_eval[i]
    else:
        U[0,i] = 2.0 - 2.0 * x_eval[i]

for k in range(0,len(t_eval) - 1):
    for i in range(1,len(x_eval) - 1):
        U[k + 1, i] = U[k,i] + ratio * (U[k, i + 1] - 2.0 * U[k, i] + U[k, i - 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(T, X, U, rstride = 10, cstride=1)
plt.title('Forward Euler Method with Delta t 0.0013')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0,0.06])
plt.ylim([0,1.0])

###Backward Euler method

##delta_t 0.005

delta_t = 0.005
t_eval = np.arange(0.0, 0.06 + delta_t, delta_t)
ratio = delta_t / (delta_x**2.0)
X, T = np.meshgrid(x_eval, t_eval)
U = np.zeros(X.shape)
for i in range(0,len(x_eval)):
    if (x_eval[i] < 0.5):
        U[0,i] = 2.0 * x_eval[i]
    else:
        U[0,i] = 2.0 - 2.0 * x_eval[i]

A = -2.0 * np.eye(len(x_eval)) + 1.0 * np.eye(len(x_eval), k=1) + 1.0 * np.eye(len(x_eval), k = -1)
A[0, :] = 0
A[:, 0] = 0
A[-1, :] = 0
A[:, -1] = 0
I = np.eye(len(x_eval))
for k in range(0, len(t_eval) - 1):
    U[k+1, :] = la.solve(I - ratio * A, U[k, :])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(T, X, U, rstride = 1, cstride=1)
plt.title('Backward Euler method with Delta t 0.005')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0, 0.06])
plt.ylim([0, 1.0])

###crank-nicolson

##delta_t 0.005

ratio = delta_t / (2.0 * (delta_x**2.0))
for k in range(0,len(t_eval)-1):
    U[k + 1, :] = la.solve(I - ratio * A, np.dot(U[k, :], I + ratio * A))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(T, X, U, rstride = 1, cstride=1)
plt.title('Crank-Nicolson with Delta t 0.005')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0, 0.06])
plt.ylim([0, 1.0])

###finite difference

ratio = 1.0 / (delta_x**2.0)

def func(y, t):
    return ratio * np.dot(A,y)

dt = 1.0e-3
y_t = np.arange(0, 0.06 + dt, dt)
y_0 = np.zeros(x_eval.shape)

for i in range(0,len(x_eval)):
    if (x_eval[i] < 0.5):
        y_0[i] = 2.0 * x_eval[i]
    else:
        y_0[i] = 2.0 - 2.0 * x_eval[i]

X, T = np.meshgrid(x_eval, y_t)
sol = integrate.odeint(func, y_0, y_t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(T, X, sol, rstride = 10, cstride=1)
plt.title('Finite Difference Method')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0,0.06])
plt.ylim([0,1.0])
plt.show()