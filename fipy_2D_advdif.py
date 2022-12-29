from fipy import CellVariable, Grid2D, ExponentialConvectionTerm, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time

def advdiff_fipy_2D(sink=True, save=False): 
    def ArrSum(array):
        ''' Computes the sum off all the elements in the 2D array
        Useful to check if the solution is conservative
        array = 2D array with a value on each value of a mesh
        '''
        return sum(map(sum,array))

    mass = []

    # Model parameters
    Ly, Lz = 30, 5
    Ny, Nz = 300, 50
    dy, dz = Ly/Ny, Lz/Nz
    v      = 0.1 
    w      = 0.01
    u      = (v,w)
    D      = (2, 0.1)

    if sink:
        wr   = (0, 0.00154)
        mode = 'sink'
    else:
        wr   = (0, - 0.00085)
        mode = 'float'

    T  = 100
    dt = T/10000
    Np = 10                                     # Number of snapshots of the solution (one every Nt/Np time steps)
    Nt = Np*np.ceil(T/(Np*dt)).astype('int')    # Nt must be an integer divisible by Np

    # Define the grid/mesh
    mesh = Grid2D(nx=Ny, ny=Nz, dx=dy, dy=dz)
    y, z = mesh.cellCenters[0], mesh.cellCenters[1]

    # Define the model variable and set the boundary conditions
    phi = CellVariable(name="numerical solution", mesh=mesh, value=numerix.exp(-(y**2 + z**2)))
    meshBnd0   = mesh.facesLeft  | mesh.facesTop
    meshBnd    = mesh.facesRight | mesh.facesBottom 
    meshAllBnd = mesh.facesLeft  | mesh.facesTop | mesh.facesRight | mesh.facesBottom 
    phi.faceGrad.constrain(0, meshBnd)         # impose zero flux on bottom and right boundaries 
    phi.constrain(0, meshBnd0)                 # impose value = 0 on top and left boundaries 

    # Define the equation
    eq = TransientTerm() == DiffusionTerm(coeff=D) \
        - ExponentialConvectionTerm(coeff=u) \
        + DiffusionTerm(coeff=wr)

    print('-- FIPY --')
    print('Time step = ' + str(round(dt,4)))
    tic = time.time()

    # Solve the equation
    my_sol = np.zeros((Np,Ny*Nz))      # Matrix with Np solution snapshots
    my_sol[0,:] = phi
    k = 1
    for step in np.arange(1,Nt):
        eq.solve(var=phi, dt=dt)
        if np.mod(step,Nt/Np)==0:
            print(step,k)
            my_sol[k,:] = phi
            k += 1
    tac = time.time()
    tictac = tac-tic
    # Plot & save the solution
    xg, yg  = np.meshgrid(np.linspace(0,Ly,Ny+1), np.linspace(0,Lz,Nz+1))
    xd, yd  = np.meshgrid(np.linspace(dy/2,Ly,Ny), np.linspace(dz/2,Lz,Nz))
    for i in np.arange(Np):
        plt.figure()
        sol = my_sol[i,:].reshape((Ny,Nz))
        sol = griddata((xd.ravel(),yd.ravel()), my_sol[i,:], (xg, yg), method='nearest')
        plt.contourf(xg, -1*yg, sol, cmap='RdBu', vmin=0) 
        plt.clim(0,1)
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel('Horizontal distance [m]')
        ax.set_ylabel('Depth [m]')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        if save:
            plt.savefig('./image_fipy/'+str(i)+mode+'.png', dpi=400)
            plt.savefig('./image_fipy/'+str(i)+mode+'.svg', format='svg')
        plt.close()
        mass.append(ArrSum(sol))


    print('Run time is ' + str(round(tictac/60,2)) + ' minutes')
    return mass


mass_sink  = advdiff_fipy_2D(sink=True, save=True)
mass_float = advdiff_fipy_2D(sink=False, save=True)


# Investigate mass conservation 
plt.figure()
plt.plot(range(10), mass_sink, '-o', label ='positive bouancy')
plt.plot(range(10), mass_float, '-.o', label='negative bouancy')
plt.xlabel('Time')
plt.ylabel('Total concentration [g/m$^3$]')
plt.legend(loc='best')
plt.grid(linestyle=':')

plt.tight_layout()
plt.savefig('./image_fipy/mass_conservation' + '.svg', format='svg', dpi=500)
plt.savefig('./image_fipy/mass_conservation' + '.png', format='png', dpi=500)
plt.close()
