import numpy as np
import matplotlib.pyplot as plt

def advdif_AB3_2D(sink=True, save=False):
	''' Solves a 2D advection-diffusion equation
		using an Adam-Bashforth 3rd order scheme
		on the domain [0,L]x[0,L]

		Based on the advection_diffusion_AB3.m script
		written by E.Hanert for the LBRTI2102 course
	''' 

	#########################
	### Useful functions ###
	#########################

	def get_grid(L, Nx, Nz):
		''' Creates the grid 
			L     = domain length
			Nx,Ny = Nbr of elements along the x/z axis
		'''
		x,y = np.meshgrid(np.linspace(0,L, Nx), np.linspace(0,L, Nz), indexing='ij')
		return x,y

	def get_advection_velocity_Z(x,y,U0,H):
		''' Computes the advection velocity on the XY plane using the hypothesis
			that advection decreases linearly with depth Z
		U0 = horinzontal velocity at the surface [m/s]
		H  = depth [m]

		Return a vector containing an advection velocity value for each node
		'''

		u = U0 - x/H
		v = U0 -y/H
		# avoid negative values
		u[u<0] = 0
		v[v<0] = 0

		return u,v

	def get_initial_conditions(y,z):
		''' Computes the initial conditions
			Plastic concentration is concentrated in the upper-left corner of the domain

		Return the initial conditions 
		'''
 
		Cinit = np.exp(-y**2 - z**2)
		return Cinit

	def ArrSum(array):
		''' Computes the sum off all the elements in the 2D array
		Useful to check if the solution is conservative

		array = 2D array with a value on each value of a mesh
		'''
		return sum(map(sum,array))

	def plot_solution(y,z,C):
		plt.figure()
		plt.contourf(y,-1*z,C)
		#plt.colorbar(fig)
		plt.show()

	# Model parameters
	# ----------------
	L  = 300               # domain length along x and y
	Us = 0.5               # amplitude of the advection velocity --> sert à quoi???
	Kv = 1e-2              # vertical diffusion coefficient
	Kh = 2                 # horinzontal diffusion coefficient
	U0 = 1                 # Surface velocity (hyp)
	w  = 1.e-4             # vertical velocity [m/s]

	if sink==True:
		wr = 0.00154       # sink velocity for PET (in regard to the Z axis) [m/s]
		mode = 'sink'
	else:
		wr = - 0.00085     # rise velocity for HDPE (in regard to the Z axis) [m/s]
		mode = 'float'

	Ny = 150               # number of elements along x (=> Nx+1 nodes)
	Nz = 150               # number of elements along y (=> Ny+1 nodes)
	Dy = L/Ny              # grid size along x
	Dz = L/Nz              # grid size along y
	T  = 100               # integration time
	dt = 0.5 * min([Dy/Us,(Dy**2)/(2*Kh),Dz/Us,(Dz**2)/(2*Kv)])
	Nt = int(10*np.ceil(T/(10*dt)))  # number of timesteps

	print('-- AB3 2D --\n')
	print('Time step = ' + str(round(dt,4)))

	# Initialization
	y,z = get_grid(L, Ny, Nz)                    # grid nodes coordinates
	v,w = get_advection_velocity_Z(y,z,U0, L)    # advection velocity for all nodes
	C   = get_initial_conditions(y,z)            # initial condition
	# print(v)
	# #print(v[0])
	# print('\n')
	# print(w)
	# print(w[0][1])
	# print('\n')
	# print(C)

	# Plot initial condtion
	plt.figure()
	plt.contourf(y,-1*z,C)
	plt.xlim(0,3)
	plt.ylim(-3,0)
	#plt.scatter(y,-1*z,C)
	plt.savefig('./images_AB3_2D/0initial_cdt.png')
	

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
		for i in range(Ny): 
			# Loop in space (z)
			for j in range(Nz):
				Cij = C[j][i]   # first get depth (j) then get value on Y axis (i)
				vij = v[j][i]   # advection velocity at a depth=j on Y axis
				wij = w[j][i]   # advection velocity at a depth=j on Z axis

				# Values around Cij + boundary conditons : No flux in/out domain
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
				difz_ij = Kv*(Cr-2*Cij+Cl)/(Dz*Dz)

				# advection (upwind)
				advy_ij = -(vij*Cij-vl*Cl)/Dy

				if w[j][i]+wr >= 0:
					advz_ij = -(wij*Cij-wd*Cd)/Dz - wr*(Cij-Cd)/Dz
				else:
					advz_ij = -(wu*Cu-wij*Cij)/Dz - wr*(Cu-Cij)/Dz

				# compute solution at node i,j
				r0[j][i] =  advy_ij + advz_ij + dify_ij + difz_ij 

		# Update solutions
		if k==1:    # FE 
			#print('FE')
			C = C + dt*r0
		elif k==2:    # AB2
			#print('AB2')
			C = C + dt*((3/2)*r0 - (1/2)*r1)
		else:    # AB3
			#print('AB3')
			C = C + dt*((23/12)*r0 - (16/12)*r1 + (5/12)*r2)

		# Print information
		if k == 1:
			print('snapshot ' + ' | ' + ' Sum')

		# Plot solution
		if np.remainder(k, Nt/10) == 0:
			snapshot = k*10//Nt
			print(str(snapshot) + '\t    ' + str(round(ArrSum(C),4)))
			plt.figure()
			plt.contourf(y,-1*z,C)
			plt.savefig('./images_AB3_2D/'+str(snapshot)+'.png')
			#plt.show()


advdif_AB3_2D(sink=True)
