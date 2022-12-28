import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
eps = np.finfo(float).eps

def upwind_adveciton_diffusion_AB3(sink=True, save=False):
    mass = []

    # Basic model parameters
    h  = 400.                          # Depth [m]
    K  = 1.e-2                         # Diffusivity [m2/s]
    Ks = 1e-1                          # Diffusivity for the upper layer (surface) [m2/s] 
    w  = 1.e-4                         # Vertical velocity [m/s]

    if sink == True:
        wr   =  0.00154               # sink velocity for PET (in regard to the Z axis) [m/s]
        mode = 'sink' 
    else:
        wr   =  - 0.00085             # rise velocity for HDPE (in regard to the Z axis) [m/s]
        mode = 'float'  

    Nz = 800                          # Number of vertical points
    dz = h/Nz                         # Space between two points
    z  = np.linspace(0,h, Nz+1)       # List of depth
    T  = 3600*24                      # Simulation time 

    dt = 0.5 * np.min([dz/np.abs(w+eps), (dz/np.abs(wr+eps)), (dz**2)/(2*K+eps), (dz**2)/(2*Ks+eps)])   # time between two iterations [s] Rem: dt < min(1/2*(dz**2)/K, 1/f)
    Nt  = int(10*np.ceil(T/(10*dt)))  # number of time steps

    print('\n-- AB3 --')
    print(f'Time step = {round(dt,4)}\nSpace step = {dz}\nTime points = {Nt}\nSpace points = {Nz}\n')

    # Initial condition
    C_0 = 1e-3                #  Concentration initiale 
    C   = np.exp(-z**2) * C_0
    r0  = np.zeros_like(C)
    r1  = np.zeros_like(C)
    r2  = np.zeros_like(C)

    # Plot & save initial condition
    plt.figure()
    plt.plot(C, -1*z, 'r')
    plt.ylabel('Depth [m]')
    plt.xlabel('Microplastic concentration [g/m$^3$]')
    plt.grid(linestyle=':')
    plt.ylim(-10,0.5)
    plt.tight_layout()
    plt.savefig('./image_AB3/inital_cdt' + '.svg', format='svg', dpi=500)
    plt.savefig('./image_AB3/inital_cdt' + '.png', format='png', dpi=500)
    plt.close()

    print('C0 = ' + str(C.sum()))

    # Initialize figure
    fig = plt.figure()
    ax  = plt.gca()

    # Loop in time
    for k in range(1, Nt+1):
        # Update solution for t-1 and t-2
        r2 = r1 ; r1 =r0
        # Loop in space
        for i in range(1, Nz-1):
            if i <= 3:
                #print(1)
                if w + wr >= 0 :
                    # No advection for the first 4 depth layers
                    r0[i] = (Ks/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) 
                else:
                    r0[i] = (Ks/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) 
            elif i <= 30//dz:
                # For the surface layer (H>=30m), diffusivity coefficient is more important 
                if w + wr >= 0 :
                    r0[i] = (Ks/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i]-C[i-1]) - (wr/dz)*(C[i]-C[i-1])
                else:
                    #print(3)
                    r0[i] = (Ks/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i+1]-C[i]) - (wr/dz)*(C[i+1]-C[i])

            else:
                #print(2)
                if w + wr >= 0 :
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i]-C[i-1]) - (wr/dz)*(C[i]-C[i-1])
                else:
                    #print(3)
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i+1]-C[i]) - (wr/dz)*(C[i+1]-C[i])

        # Update C
        if k==1:    # Forward Euler
            C = C + dt*r0
        elif k==2:
            C = C + dt*((3/2)*r0 - (1/2)*r1)
        else:
            C = C + dt*((23/12)*r0 - (16/12)*r1 + (5/12)*r2)

        # Boundary condtition : no flux in/out domain
        C[0]  = 0 
        C[-1] = C[-2]
        # Plot the solution
        if np.remainder(k, Nt/10) == 0:
            ax.plot(C, -1*z)
            snapshot = k*10//Nt
            print(str(snapshot) + '\t' + str(C.sum()))
            mass.append(C.sum())     

    # Display result
    plt.grid(linestyle=':')
    plt.ylabel('Depth [m]')
    plt.xlabel('Microplastic concentration [g/m$^3$]')
    #plt.title(mode)
    plt.tight_layout()

    if save==True:
        plt.savefig('./image_AB3/AB3_upwind_1D_' + mode + '.svg', format='svg', dpi=500)
        plt.savefig('./image_AB3/AB3_upwind_1D_' + mode + '.png', format='png', dpi=500)

    plt.show()

    return mass


# Simulation for negative buoyancy plastic
float_mass = upwind_adveciton_diffusion_AB3(sink=True, save=False)

# Simulation for positive buoyancy plastic
sink_mass = upwind_adveciton_diffusion_AB3(sink=False, save=False)


# Investigate mass conservation 
plt.figure()
plt.plot(range(10), float_mass, '-o', label ='positive bouancy')
plt.plot(range(10), sink_mass, '-o', label='negative bouancy')
plt.xlabel('Time')
plt.ylabel('Total concentration [g/m$^3$]')
plt.legend(loc='best')
plt.grid(linestyle=':')

plt.tight_layout()
plt.savefig('./image_AB3/mass_conservation' + '.svg', format='svg', dpi=500)
plt.savefig('./image_AB3/mass_conservation' + '.png', format='png', dpi=500)
plt.show()

