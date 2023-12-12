"""
heatequation.py
Name(s): Lilia Qian
NetId(s): lyq2
Date: due 12/13/2023
"""

import numpy as np
import matplotlib.pyplot as plt


"""
solveheatequation: solves the heat equation numerically using the Crank-Nicolson method and plots the result

INPUTS
L: floating point, length of rod
T: floating point, total simulation time
dx: floating point, step size along x-axis
dt: floating point, step size along t-axis
cond0: floating point, x=0 boundary condition
condL: floating point, x=L boundary condition
bc: string, specifies type of boundary condition
alpha2: floating point, thermal diffusivity of the rod

OUTPUTS
A plot showing the temperature distribution in the rod at each timestep

"""
def solveheatequation(L=1.0, T=0.1, dx=0.1, dt=0.025, cond0=0.0, condL=0.0, bc='dirichlet', alpha2 =1.0):

    # fill an array with x-step values e.g. [0, 0.1, 0.2, ... , 0.9, 1.0]
    x = np.arange(0, L + dx, dx)  # go from x = 0 to x = L (rod length is L)
    # fill an array with t-step values e.g. [0, 0.025, 0.05, ... , 0.1]
    t = np.arange(0, T + dt, dt).round(3)  # go from t = 0 to t = T (total time interval T)

    if bc == 'dirichlet':
        boundaryConditions = [cond0, condL]
    elif bc == 'neumann':
        boundaryConditions = [cond0*dx, condL*dx]

    initialConditions = np.sin(np.pi * x)  # u(x,0) = [sin(0), sin(pi*x[1]), sin(pi*x[2]), ... , sin(pi*x[-1])]

    # count the number of steps in x-direction
    n = len(x)
    # count the number of steps in the t-direction
    m = len(t)
    # preallocate a matrix to store temperature results with dimensions n by m (x-steps by t-steps)
    u = np.zeros((n, m))

    # Fill the temperature result matrix with the initial conditions
    u[0, :] = boundaryConditions[0]
    u[-1, :] = boundaryConditions[1]
    # At time 0, fill every x value with the specified initial conditions
    u[:, 0] = initialConditions

    # calculate lambda
    lam = alpha2 * dt/dx**2

    # Make the matrices needed in the solution to the heat equation
    A = np.diag([2 + 2 * lam] * (n - 2), 0) + np.diag([-lam] * (n - 3), -1) + np.diag([-lam] * (n - 3), 1)
    # print(A)
    B = np.diag([2 - 2 * lam] * (n - 2), 0) + np.diag([lam] * (n - 3), -1) + np.diag([lam] * (n - 3), 1)
    # print(B)

    # loop through time (applying Crank-Nicolson)
    for j in range(0, m - 1):
        b = u[1:-1, j].copy()  # a vector containing the initial conditions
        b = B @ b
        b[0] = b[0] + lam * (u[0, j] + u[0, j + 1])
        b[-1] = b[-1] + lam * (u[-1, j] + u[-1, j + 1])

        # Update the Neumann boundary conditions
        b[0] = b[0] + 2 * lam * boundaryConditions[0]
        b[-1] = b[-1] + 2 * lam * boundaryConditions[1]

        solution = np.linalg.solve(A, b)
        u[1:-1, j + 1] = solution # update T matrix with solved b values

    print(u.round(2))  # print out the solution matrix

    # Now plot the solution
    R = np.linspace(1,0,m)
    B = np.linspace(0,1,m)
    G = 0

    for j in range(m):
        plt.plot(x, u[:,j], color = [R[j], G, B[j]])

    plt.suptitle('Distance v. Temperature')
    plt.title('Boundary conditions: ' + bc + '; Alpha-squared: ' + str(alpha2))
    plt.xlabel('distance [m]')
    plt.ylabel('Temperature [$\degree$ C]')
    plt.legend(t)
    plt.show()


if __name__ == '__main__':
    #######################################################
    # Experiments on rod material
    #######################################################

    # Use a standard rod as a base case (default values)
    # L = 1.0, T = 0.1 , dx = 0.1, dt = 0.025, cond0 = 0, condL = 0, bc = 'dirichlet', alpha2 = 1.0
    solveheatequation()

    # Modify the material of the base rod to be 99.9% pure silver
    solveheatequation(alpha2 = 1.6563)

    # Modify the material of the base rod to be air at 300K
    solveheatequation(alpha2 = 0.19)

    # Modify the material of the base rod to be water
    solveheatequation(alpha2 = 0.00144)

    #######################################################
    # Experiments on rod length
    #######################################################

    solveheatequation(L = 1.0)
    solveheatequation(L = 2.0)
    solveheatequation(L = 4.0)
    solveheatequation(L = 10.0)

    #######################################################
    # Experiment with various dirichlet boundary conditions
    #######################################################

    solveheatequation()
    solveheatequation(cond0 = 50.0, condL = 0.0)
    solveheatequation(cond0 = 50.0, condL = 50.0)
    solveheatequation(cond0 = -50.0, condL = -50.0)

    #######################################################
    # Experiment with various neumann boundary conditions
    #######################################################

    solveheatequation(cond0 = -0.5, condL = -0.5, bc="neumann")
    solveheatequation(cond0 = -2.0, condL = -2.0, bc="neumann")
    solveheatequation(cond0 = -5.0, condL = -5.0, bc="neumann")
