import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

def simple_adv_diff_FE(sink=True):
	# Basic model parameters
	h = 1         # [m]
	K = 1e-9         # vertical diffusivity [m2/s] 
	u = 0              # no advection when solving only for z
	w = 0.01    # vertical velocity [m/s]

	if sink == True:
		wr = -0.00154 # sink velocity for PET [m/s]
	else:
		wr = 0.00085  # rise velocity for HDPE [m/s]

	Nz = 1000                      # number of vertical layers
	dz = h/Nz                        # space between two layers [m]
	z = np.linspace(0, -h, Nz+1)     # list of depth, vertical grid to compute C
	T = 100 #3600*24#*31*12*2              # simulation time - first try = 1 week
	
	dt = np.min([ dz/np.abs(w+eps), (dz**2)/(2*K)])


	if dt > np.min([ dz/np.abs(w+eps), (dz**2)/(2*K)]):      # time between two iterations [s] Rem: dt < min(1/2*(dz**2)/K, 1/f)
		print('warning: dt does not satisfy the stability conditon!')
	Nt = int(10*np.ceil(T/(10*dt))) # number of time steps

	nu = dt*(-w-wr)/(2*dz)
	mu = K*dt/(dz**2)

	print((dz**2)/(2*K+eps))

	print('\n-- Forward Euleur --')
	print(f'Time step = {dt}\nSpace step = {dz}\nTime points = {Nt}\nSpace points = {Nz}\n')


	# Iitialize figure
	fig = plt.figure()
	ax = plt.gca()

	#initial condition
	Cold = np.exp(-z**2)
	#Cold = np.zeros(Nz+1) + 0.5
	Cnew = Cold
	#print(np.size(Cnew))
	#print(np.size(z))
	ax.plot(Cold, z, 'r')

	# Loop in time
	for k in range(1, Nt+1):
		# Loop in space
		for i in range(1, Nz-1):
			#Cnew[i] = Cold[i] + mu*(Cold[i+1]-2*Cold[i]-Cold[i-1]) + nu*(Cold[i+1]-Cold[i-1])
			Cnew[i] = Cold[i] + mu*(Cold[i] - Cold[i-1]) + nu*(Cold[i]-Cold[i-1])
		# Boundary condition : no flux in/out domain
		Cnew[Nz] = Cnew[Nz-1]
		print(Cnew[0])
		Cnew[0] = Cnew[1]
		print(Cnew[0])

		# Plot the solution
		if np.remainder(k, Nt/10) == 0:
			ax.plot(Cnew, z, 'k')


		#print(Cnew)

	# Show the result
	print(nu)
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.grid(linestyle=':')
	plt.savefig('./image/plastic_cc.png', dpi=400)
	plt.show()


simple_adv_diff_FE(sink=True)





