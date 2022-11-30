import matplotlib.pyplot as plt
import numpy as np
eps = np.finfo(float).eps

## Equation to solve : dC/dt = - w dC/dz        - wr dC/dz          + K d^2C/dz^2 
#					          [- advection water - advection plastic + diffusion water)]
# C = microplastic concentration [g/m^3]
# w, wr, K = constante


def upwind_adveciton_diffusion_FE(sink=True , svg=False, save=False):
	# Basic model parameters
	h = 100.                           # Depth [m]
	K = 1.e-4                         # Diffusivity [m2/s] - 
	w = 1.e-3                         # Vertical velocity [m/s]

	if sink == True:
		wr   =  0.00154#*10           # sink velocity for PET (in regard to the Z axis) [m/s]
		mode = 'sink' 
	else:
		wr   =  - 0.00085#*10          # rise velocity for HDPE (in regard to the Z axis) [m/s]
		mode = 'float'  

	Nz = 300                          # Number of vertical points
	dz = h/Nz                         # Space between two points
	z  = np.linspace(0,h, Nz+1)       # List of depth
	T  = 500.                         # Simulation time 

	dt = np.min([dz/np.abs(w+eps), (dz/np.abs(wr+eps)), (dz**2)/(2*K+eps)])#*0.75   # time between two iterations [s] Rem: dt < min(1/2*(dz**2)/K, 1/f)
	if dt > np.min([ dz/np.abs(w+eps), (dz/np.abs(wr+eps)), (dz**2)/(2*K)]):      
		print('warning: dt does not satisfy the stability conditon!')

	Nt  = int(10*np.ceil(T/(10*dt)))  # number of time steps

	nu  = w*dt/dz                     # Water advection coefficient
	nur = wr*dt/dz                    # Plastic advection coefficient
	mu  = K*dt/(dz**2)                # Water diffusivity coefficient


	print(np.min([ dz/np.abs(w+eps), dz/np.abs(wr+eps), (dz**2)/(2*K)]))
	print('\n-- Forward Euleur --')
	print(f'Time step = {dt}\nSpace step = {dz}\nTime points = {Nt}\nSpace points = {Nz}\n')

	# Initialize figure
	fig = plt.figure()
	ax  = plt.gca()

	# Initial condition
	C_0 = 1.077  # = Concentration initiale 
	Cold = np.exp(-z**2)*C_0 #gaussienne donc C élevée à la surface puis diminue avec la profondeur
	Cnew = Cold

	ax.plot(Cold, -1*z, 'r')
	plt.grid(linestyle=':')

	# Loop in time
	for k in range(1, Nt+1):
		# Loop in space
		for i in range(1, Nz-1):
			if w >= 0 :
				#upwind
				Cnew[i] = Cold[i] + mu*(Cold[i+1]-2*Cold[i]+Cold[i-1]) \
						          - (nu)*(Cold[i]-Cold[i-1])  \
						          - (nur)*(Cold[i]-Cold[i-1])
			elif w==0:
				wr = 0 
				#centered in space ->  no advection
				Cnew[i] = Cold[i] + mu*(Cold[i+1]-2*Cold[i]+Cold[i-1])  \
							      - (nu/2)*(Cold[i+1]-Cold[i-1]) \
							      - (nur/2)*(Cold[i+1]-Cold[i-1]) 
			else:
				#upwind
				Cnew[i] = Cold[i] + mu*(Cold[i+1]-2*Cold[i]+Cold[i-1]) \
						          - (nu)*(Cold[i+1]-Cold[i])  \
						          - (nur)*(Cold[i+1]-Cold[i]) 


		# Boundary condtition : no flux in/out domain
		Cnew[0] = Cnew [1]
		Cnew[-1] = Cnew[-2]
		# Plot the solution
		if np.remainder(k, Nt/10) == 0:
			ax.plot(Cnew, -1*z)
			print(k)

	# Display & save result
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.ylabel('Depth [m]')
	plt.xlabel('Microplastic concentration [g/m$^3$]')
	plt.title(mode)
	plt.tight_layout()

	if save==True:
		if svg==True:
			plt.savefig('./image_FE/FE_upwind_' + mode + '_' + str(int(h)) + 'm' + '.svg', dpi=300)
		else:
			plt.savefig('./image_FE/FE_upwind_' + mode + '_' + str(int(h)) + 'm' + '.png', dpi=300)
	else:
		#plt.savefig('./image_FE/plastic_cc' + str(int(h)) + '.png', dpi=300)
		pass


	plt.show()

# Simulation for negative buoyancy plastic
upwind_adveciton_diffusion_FE(sink=True, save=True)

# Simulation for positive buoyancy plastic
upwind_adveciton_diffusion_FE(sink=False, save=True)


