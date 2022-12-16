import numpy as np
import matplotlib.pyplot as plt

def advdif_AB3_2D(sink=True, save=False):
	''' Solves a 2D advection-diffusion equation
		using an Adam_bashforth 3rd oder sheme
		on the domain [0,L]x[0,L]

		Based on the advection_diffusion_AB3.m script
		writen by E.Hanert for the LBRTI2102 course
	''' 

	#########################
	### Usefull functions ###
	#########################

	def get_grid(L, Nx, Nz):
		''' Creates the grid 
			L     = domain length
			Nx,Ny = Nbr of elements along the x/z axis
		'''
		x,y = np.meshgrid(np.linspace(0,L+1, Nx//L), np.linspace(0,L+1, Nz//L), indexing='ij')
		return x,y

	def get_advection_velocity_Z(U0, H):
		''' Computes the advection velocity on the XY plane using the hypothesis
			that advection decreases linearly with depth Z

		Return a vector containing an advection velocity value for each node
		'''

		Z = np.linspace(0,L, Nz//L)
		print(Z)
		u = U0 - Z/H 
		u[u<0] = 0
		return u

	def get_initial_conditions(x,z):
		''' Computes the initial conditions
			Plastic concentration is concentrated in the upper-left corner of the domain

		Return the initial conditions [type:float64]
		'''

		#Xmin  = np.amin(x) 
		#Zmin  = np.amin(z) 
		Cinit = np.exp(-x**2 - z**2)
		return Cinit

	def plot_solution(y,z,C):
		plt.figure()
		plt.contourf(y,-1*z,C)
		#plt.colorbar(fig)
		plt.show()

	# Model parameters
	# ----------------
	L  = 3                 # domain length along x and y
	Us = 0.5               # amplitude of the advection velocity --> sert à quoi???
	Kv = 1e-2              # vertical diffusion coefficient
	Kh = 2                 # horinzontal diffusion coefficient
	U0 = 0.5               # Surface velocity (hyp)
	w  = 1.e-4             # vertical velocity [m/s]

	if sink==True:
		wr = 0.00154       # sink velocity for PET (in regard to the Z axis) [m/s]
		mode = 'sink'
	else:
		wr = - 0.00085     # rise velocity for HDPE (in regard to the Z axis) [m/s]
		mode = 'float'

	Ny = 50                # number of elements along x (=> Nx+1 nodes)
	Nz = 50                # number of elements along y (=> Ny+1 nodes)
	Dy = L/Ny              # grid size along x
	Dz = L/Nz              # grid size along y
	T  = 100               # integration time
	dt = 0.1 * min([Dy/Us,(Dy**2)/(2*Kh),Dz/Us,(Dz**2)/(2*Kv)])
	Nt = int(10*np.ceil(T/(10*dt)))  # number of timesteps

	print('-- AB3 2D --')

	# Initialization
	y,z = get_grid(L, Ny, Nz)                 # grid nodes coordinates
	u   = get_advection_velocity_Z(U0, L)     # advection velocity for each depth layer
	C  = get_initial_conditions(y,z)
	print(u)
	# print('\n')
	# print(C)
	# print('\n')
	# print(C[0])
	# print('\n')
	# print(C[0][0])

	# Plot initial condtion
	plt.figure()
	plt.contourf(y,-1*z,C)
	#plt.scatter(y,-1*z,C)
	plt.show()

	# Integration of the equation
	# First solution for time step t, t-1 and t-2
	r0  = np.zeros_like(C)
	r1  = np.zeros_like(C)
	r2  = np.zeros_like(C)

	# Loop in time
	for k in range(1, Nt+1): #when k==0 -> initial cdt! 
		# Update solution for t-1 and t-2
		r2=r1 ; r1=r0

		# Loop in space (y)
		for i in range(Ny+1): #si pas de BC à droite: range(1, Ny)?? 
			# Loop in space (z)
			for j in range(Nz+1):
				Cij = C[j][i] # first get depth j then get value on Y axis
				uz = u[j]
				# Values around Cij + boundary conditons : No flux in/out domain
				if i > 0:
					Cl = C[j][i-1]  # left value
				else:
					Cl = Cij

				if i < Ny:
					Cr = C[j][i+1]  # right value
				else: 
					Cr = Cij

				if j > 0:
					Cd = C[j-1][i]  # "down" value
				else:
					Cd = Cij

				if j < Nz:
					Cu = C[j+1][i]  # "up" value
				else:
					Cu = Cij

				# diffusion (centered)
				dify_ij = Kh*(Cr-2*Cij+Cl)/(Dy*Dy)
				difz_ij = Kv*(Cr-2*Cij+Cl)/(Dz*Dz)

				# advection (upwind)
				advy_ij = - u[j]*(Cij-Cl)/Dy

				if w+wr >= 0:
					advz_ij = - w*(Cij-Cd)/Dz - wr*(Cij-Cd)/Dz
				else:
					advz_ij = - w*(Cu-Cij)/Dz - wr*(Cu-Cij)/Dz

				# compute solution at node i,j
				r0[j][i] = advy_ij + advy_ij + dify_ij + difz_ij

		# Update solutions
		if k==1:    # FE 
			C = C + dt*r0
		if k==2:    # AB2
			C = C + dt*((3/2)*r0 - (1/2)*r1)
		if k==3:
			C = C + dt*((23/12)*r0 - (16/12)*r1 + (5/12)*r2)

		# Plot solution
		if np.remainder(k, Nt/10) == 0:
			plt.figure()
			plt.contourf(y,-1*z,C)
			plt.show()












# get_advection_velocity_Z(0.5,50)
# x,z = get_grid(10, 50, 50)
# get_initial_conditions(x,z)

advdif_AB3_2D()
