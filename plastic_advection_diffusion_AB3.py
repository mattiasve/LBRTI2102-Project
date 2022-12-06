import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
eps = np.finfo(float).eps

def upwind_adveciton_diffusion_AB3(sink=True, save=False):
    all_C = []
    sum_C = []

    # Basic model parameters
    h = 100.                           # Depth [m]
    K = 1.e-2                         # Diffusivity [m2/s]  
    w = 1.e-4                         # Vertical velocity [m/s]

    if sink == True:
        wr   =  0.00154               # sink velocity for PET (in regard to the Z axis) [m/s]
        mode = 'sink' 
    else:
        wr   =  - 0.00085             # rise velocity for HDPE (in regard to the Z axis) [m/s]
        mode = 'float'  

    Nz = 1000                          # Number of vertical points
    dz = h/Nz                         # Space between two points
    z  = np.linspace(0,h, Nz+1)       # List of depth
    T  = 3600                         # Simulation time 

    dt = 0.5 * np.min([dz/np.abs(w+eps), (dz/np.abs(wr+eps)), (dz**2)/(2*K+eps)])   # time between two iterations [s] Rem: dt < min(1/2*(dz**2)/K, 1/f)
    Nt  = int(10*np.ceil(T/(10*dt)))  # number of time steps

    print('\n-- AB3 --')
    print(f'Time step = {round(dt,4)}\nSpace step = {dz}\nTime points = {Nt}\nSpace points = {Nz}\n')

    # Initialize figure
    fig = plt.figure()
    ax  = plt.gca()

    # Initial condition
    C_0 = 1.077          #  Concentration initiale 
    C   = np.exp(-z**2) * C_0
    r0  = np.zeros_like(C)
    r1  = np.zeros_like(C)
    r2  = np.zeros_like(C)

    ax.plot(C, -1*z, 'r')
    print('C_0 = ' + str(C.sum()))

    # Store all values for countour plot
    C_all_sol = pd.DataFrame(index=range(Nt+1),columns=range(Nz+1))
    C_all_sol.iloc[0] = C

    # Loop in time
    for k in range(1, Nt+1):
        # Update solution for t-1 and t-2
        r2 = r1 ; r1 =r0
        # Loop in space
        for i in range(1, Nz-1):
            if i <= 3:
                #print(1)
                w  = 0  
                wr = 0
                if w + wr >= 0 :
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i]-C[i-1]) - (wr/dz)*(C[i]-C[i-1])
                else:
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i+1]-C[i]) - (wr/dz)*(C[i+1]-C[i])
            else:
                #print(2)
                if w + wr >= 0 :
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i]-C[i-1]) - (wr/dz)*(C[i]-C[i-1])
                else:
                    r0[i] = (K/(dz**2))*(C[i+1]-2*C[i]+C[i-1]) - (w/dz)*(C[i+1]-C[i]) - (wr/dz)*(C[i+1]-C[i])

        # Update C
        if k==1:    # Forward Euler
            C = C + dt*r0
        elif k==2:
            C = C + dt*((3/2)*r0 - (1/2)*r1)
        else:
            C = C + dt*((23/12)*r0 - (16/12)*r1 + (5/12)*r2)

        # Boundary condtition : no flux in/out domain
        C[0]  = C[1]
        C[-1] = C[-2]
        # Plot the solution
        if np.remainder(k, Nt/10) == 0:
            ax.plot(C, -1*z)
            print(C.sum())
            all_C.append(C)
            sum_C.append(C.sum())

        # Store all solutions
        C_all_sol.iloc[k] = C

    print(C_all_sol)

    # Display result
    plt.grid(linestyle=':')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylabel('Depth [m]')
    plt.xlabel('Microplastic concentration [g/m$^3$]')
    #plt.title(mode)
    plt.tight_layout()

    if save==True:
        plt.savefig('./image_AB3/AB3_upwind_1D_' + mode + '.svg', format='svg', dpi=500)
        plt.savefig('./image_AB3/AB3_upwind_1D_' + mode + '.png', format='png', dpi=500)

    plt.show()

    return all_C, sum_C, z

# Simulation for negative buoyancy plastic
sink_cc, sink_sum, depth   = upwind_adveciton_diffusion_AB3(sink=True, save=True)

# Simulation for positive buoyancy plastic
float_cc, float_sum, depth = upwind_adveciton_diffusion_AB3(sink=False, save=True)

