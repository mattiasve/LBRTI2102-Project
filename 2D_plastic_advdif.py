import numpy as np
import matplotlib.pyplot as plt
import time

def advdif_AB3_2D(sink=True, save=False):
	''' Solves a 2D advection-diffusion equation
		using an Adam-Bashforth 3rd order scheme
		on the domain [0,L]x[0,L]

		Based on the "advection_diffusion_AB3.m" script
		written by E.Hanert for the LBRTI2102 course
	''' 

	#########################
	### Useful functions ####
	#########################

	def get_grid(Ly, Lz, Ny, Nz):
		''' Creates the grid 
			L     = domain length
			Nx,Nyz = Nbr of elements along the x/z axis
		'''
		y,z = np.meshgrid(np.linspace(0,Ly, Ny), np.linspace(0,Lz, Nz), indexing='xy')
		return y,z

	def get_advection_velocity_Z(y,z,U0,H):
		''' Computes the advection velocity using the hypothesis
			that advection decreases linearly with depth Z
		U0 = horinzontal velocity at the surface [m/s]
		H  = depth [m]

		Return a vector containing an advection velocity value for each node
		'''
		v = U0 - z/H
		w = (U0 - z/H)*1e-4
		# avoid negative values
		w[w<0] = 0
		v[v<0] = 0

		return v,w

	def get_initial_conditions(y,z,C0=1e-3):
		''' Computes the initial conditions : Gaussian function
			Plastic concentration is concentrated in the upper-left corner of the domain

		Return the initial conditions 
		'''
		Cinit = np.exp(-y**2 - z**2)*C0
		return Cinit

	def ArrSum(array):
		''' Computes the sum off all the elements in the 2D array
		Useful to check if the solution is conservative

		array = 2D array with a value on each value of a mesh
		'''
		return sum(map(sum,array))

	# Model parameters
	# ----------------
	Ly = 2000              # domain length along Y
	Lz = 30                # domain length along Z
	Us = 0.5               # amplitude of the advection velocity (?)
	Kv = 1e-2              # vertical diffusion coefficient
	Ks = 1e-1              # vertical diffusivity for the upper layer
	Kh = 2                 # horinzontal diffusion coefficient
	U0 = 0.5               # Surface velocity (hyp)

	if sink==True:
		wr   = 0.00154     # sink velocity for PET  (in regard to the Z axis) [m/s]
		mode = 'sink'
	else:
		wr   = - 0.00085   # rise velocity for HDPE (in regard to the Z axis) [m/s]
		mode = 'float'

	Ny = 1000               # number of elements along y 
	Nz = 30                 # number of elements along z 
	Dy = Ly/Ny              # grid size along y
	Dz = Lz/Nz              # grid size along z
	T  = 3600               # integration time
	dt = 0.5 * min([Dy/Us,(Dy**2)/(2*Kh),Dz/Us,(Dz**2)/(2*Kv)])
	Nt = int(10*np.ceil(T/(10*dt)))  # number of timesteps
	mass = []

	print('-- AB3 2D --\n')
	print('Time step = ' + str(round(dt,4)))

	# Initialization
	y,z = get_grid(Ly, Lz, Ny, Nz)                # grid nodes coordinates
	v,w = get_advection_velocity_Z(y,z,U0, Lz)    # advection velocity for all nodes
	C   = get_initial_conditions(y,z)             # initial condition

	# Plot & save initial condtion
	plt.figure()
	plt.contourf(y,-1*z,C, cmap='RdBu')
	plt.xlim(0,3)
	plt.ylim(-3,0)
	plt.colorbar()
	ax=plt.gca() ; ax.xaxis.tick_top()
	plt.xlabel('Horizontal distance [m]')
	plt.ylabel('Depth [m]')
	ax.xaxis.set_label_position('top') 

	if save:
		plt.savefig('./images_AB3_2D/0initial_cdt.png', dpi=500)
		plt.savefig('./images_AB3_2D/0initial_cdt.svg', dpi=500, format='svg')
	plt.close() 
	
	#####################################
	###  Integration of the equation  ###
	#####################################

	# First solution for time step t, t-1 and t-2
	r0  = np.zeros_like(C)
	r1  = np.zeros_like(C)
	r2  = np.zeros_like(C)

	tic = time.time() 
	# Loop in time
	for k in range(1, Nt+1): 
		# Update solution for t-1 and t-2
		r2=r1 ; r1=r0

		# Loop in space (y)
		for i in range(Ny): 
			# Loop in space (z)
			for j in range(Nz):
				Cij = C[j][i]     # first get depth (j) then get value on Y axis (i)
				vij = v[j][i]     # advection velocity at a depth=j on Y axis
				wij = w[j][i]     # advection velocity at a depth=j on Z axis

				# Values around Cij + boundary conditons : 
				#   No flux in/out domain for bottom layer and right side
				#   C = 0 for top layer and left side (avoid non-conservation mass issues)
				if i > 0:
					Cl = C[j][i-1] ; vl = v[j][i-1]  # left value
				else:
					Cl = 0 ; vl = vij

				if i < Ny-1:
					Cr = C[j][i+1] ; vr = v[j][i+1] # right value
				else: 
					Cr = Cij ; vr = vij

				if j > 0:
					Cd = C[j-1][i] ; wd = w[j-1][i] # "down" value
				else:
					Cd = 0 ; wd = wij

				if j < Nz-1:
					Cu = C[j+1][i] ; wu = w[j+1][i] # "up" value
				else:
					Cu = Cij ; wu = wij

				# diffusion (centered)	 
				dify_ij = Kh*(Cr-2*Cij+Cl)/(Dy*Dy)
				## for the surface layer (H>=30m), diffusivity coefficient is more important
				if j <= 30//Dz : 
					difz_ij = Ks*(Cr-2*Cij+Cl)/(Dz*Dz)
				else:
					difz_ij = Kv*(Cr-2*Cij+Cl)/(Dz*Dz)

				# advection (upwind)
				## advection on Y is always >0
				advy_ij = -(vij*Cij-vl*Cl)/Dy

				if w[j][i]+wr >= 0:
					advz_ij = -((wij*Cij-wd*Cd)/Dz + wr*(Cij-Cd)/Dz)
				else:
					advz_ij = -((wu*Cu-wij*Cij)/Dz + wr*(Cu-Cij)/Dz)

				# compute solution at node i,j
				r0[j][i] =  advy_ij + advz_ij + dify_ij + difz_ij 

		# Update solutions
		if k==1:    # FE 
			C = C + dt*r0
		elif k==2:  # AB2
			C = C + dt*((3/2)*r0 - (1/2)*r1)
		else:       # AB3
			C = C + dt*((23/12)*r0 - (16/12)*r1 + (5/12)*r2)

		# Print information
		if k == 1:
			print('snapshot ' + ' | ' + ' Sum')

		# Plot solution
		if np.remainder(k, Nt/10) == 0:
			snapshot = k*10//Nt
			mass.append(ArrSum(C))
			print(str(snapshot) + '\t    ' + str(ArrSum(C)))
			plt.figure(figsize=(8,4))
			plt.contourf(y,-1*z,C, vmin=0, cmap='RdBu')
			plt.colorbar()
			plt.xlabel('Horizontal distance [m]')
			plt.ylabel('Depth [m]')
			ax=plt.gca() 
			ax.xaxis.tick_top()
			ax.xaxis.set_label_position('top') 
			if sink == False : plt.ylim(-6,0)
			if save:
				plt.savefig('./images_AB3_2D/'+str(snapshot) + mode +'.png', dpi=500)
				plt.savefig('./images_AB3_2D/'+str(snapshot) + mode +'.svg', dpi=500, format='svg')

	tac = time.time()
	tictac = tac-tic
	print('Run time is ' + str(round(tictac/60,2)) +' minutes' )
	return mass


# Simulation for sinking plastic
sink_mass  = advdif_AB3_2D(sink=True, save=True)

# Simulation for floating plastic
float_mass = advdif_AB3_2D(sink=False, save=True)

# Investigate mass conservation 
plt.figure()
plt.plot(range(10), float_mass, '-o', label ='positive bouancy')
plt.plot(range(10), sink_mass, '-o', label='negative bouancy')
plt.xlabel('Time')
plt.ylabel('Total concentration [g/m$^3$]')
plt.legend(loc='best')
plt.grid(linestyle=':')

plt.tight_layout()
plt.savefig('./image_AB3_2D/mass_conservation' + '.svg', format='svg', dpi=500)
plt.savefig('./image_AB3_2D/mass_conservation' + '.png', format='png', dpi=500)
plt.close()