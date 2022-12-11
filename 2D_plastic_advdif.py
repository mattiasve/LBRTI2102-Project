import numpy as np
import matplotlib.pyplot as plt

def advdif_AB3_2D():
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

		Z = np.linspace(0,L+1, Nz//L)
		u = U0 - Z/H
		return u

	def get_initial_conditions(x,z):
		''' Computes the initial conditions
			Plastic concentration is concentrated in the upper-left corner of the domain

		Return the initial conditions [type:float64]
		'''

		Xmin  = np.amin(x) 
		Zmin  = np.amin(z) 
		Cinit = np.exp(-Xmin**2 - Zmin**2)
		return Cinit

	def plot_solution(x,z,C,k):
		if k ==1:
			ax = plt.subplot(3,3,1)
			ax.contour(x,z,C)
			plt.show()
		else:
			ax.contour(x,z,C)

		plt.show()
		return ax

	# Model parameters
	# ----------------
	L  = 10                # domain length along x and y
	Us = 0.5               # amplitude of the advection velocity --> sert à quoi???
	Kv  = 1e-2             # vertical diffusion coefficient
	Kh = 2                 # horinzontal diffusion coefficient
	U0 = 0.5               # Surface velocity (hyp)
	Nx = 50                # number of elements along x (=> Nx+1 nodes)
	Nz = 50                # number of elements along y (=> Ny+1 nodes)
	Dx = L/Nx              # grid size along x
	Dz = L/Nz              # grid size along y
	T  = 100               # integration time
	dt = 0.1 * min([Dx/Us,(Dx**2)/(2*Kh),Dz/Us,(Dz**2)/(2*Kv)])
	Nt = 8*np.ceil(T/(8*dt))  # number of timesteps

	print('--- AB3 2D ---')

	# Initialization
	x,z = get_grid(L, Nx, Nz)                 # grid nodes coordinates
	u   = get_advection_velocity_Z(U0, L)     # advection velocity for each depth layer
	C0  = get_initial_conditions(x,z)
	fig1 = plot_solution(x,z,C0,1)

	plt.figure()
	plt.contour(x,z,C0)
	plt.show()





# get_advection_velocity_Z(0.5,50)
# x,z = get_grid(10, 50, 50)
# get_initial_conditions(x,z)

advdif_AB3_2D()
