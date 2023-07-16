import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
import random as rd
import scipy.linalg as la


def linear_scattering(x,y,z,m,n,o,a,b,L,N):
	hx = L/m 
	hy = a/n
	hz = b/o
	mesh = np.zeros((m+1,n+1,o+1))

	for i in range(N):
		xgreater = False
		ygreater = False
		zgreater = False
		xsteps = 0
		ysteps = 0
		zsteps = 0
		while xgreater == False:
			if x[i]<hx*xsteps:
				xgreater = True
			else:
				xsteps += 1
		while ygreater == False:
			if y[i]<hy*ysteps:
				ygreater = True	
			else:
				ysteps += 1
		while zgreater == False:
			if z[i]<hz*zsteps:
				zgreater = True	
			else:
				zsteps += 1
		
	
		weight_nextx = (1-(hx*xsteps-x[i])/hx)
		weight_prevx = 1-weight_nextx
		weight_nexty = (1-(hy*ysteps-y[i])/hy)
		weight_prevy = 1-weight_nexty
		weight_nextz = (1-(hz*zsteps-z[i])/hz)
		weight_prevz = 1-weight_nextz

		mesh[xsteps-1][ysteps-1][zsteps-1] += weight_prevx*weight_prevy*weight_prevz
		mesh[xsteps-1][ysteps-1][zsteps] += weight_prevx*weight_prevy*weight_nextz
		mesh[xsteps-1][ysteps][zsteps-1] += weight_prevx*weight_nexty*weight_prevz
		mesh[xsteps-1][ysteps][zsteps] += weight_prevx*weight_nexty*weight_nextz
		mesh[xsteps][ysteps-1][zsteps-1] += weight_nextx*weight_prevy*weight_prevz
		mesh[xsteps][ysteps-1][zsteps] += weight_nextx*weight_prevy*weight_nextz
		mesh[xsteps][ysteps][zsteps-1] += weight_nextx*weight_nexty*weight_prevz
		mesh[xsteps][ysteps][zsteps] += weight_nextx*weight_nexty*weight_nextz
		
	return mesh




def solve_field(mesh, m,n,o,a,b,L):
	hx = L/m 
	hy = a/n
	hz = b/o
	coeff_matrix = np.zeros(((m+1)*(n+1)*(o+1), (m+1)*(n+1)*(o+1)))
	for i in range((m+1)*(n+1)*(o+1)):
		coeff_matrix[i][i] = -(2/hx**2 + 2/hy**2 + 2/hz**2)
		if ((n+1)-i%(n+1)!=1): 
			coeff_matrix[i][i+1] = 1/hy**2 #y + hy
		if (i%(n+1)!=0):
			coeff_matrix[i][i-1] = 1/hy**2 #y - hy

		if ((o+1)-(i//(n+1)%(o+1)) != 1 ): 
			coeff_matrix[i][i+(n+1)] = -(2/hx**2 + 2/hy**2 + 2/hz**2) #z + hz
		if ((i//(n+1)%(o+1)) != 0 ):
			coeff_matrix[i][i-(n+1)] = -(2/hx**2 + 2/hy**2 + 2/hz**2) #z - hz

		if ((m+1)-(i//((n+1)*(o+1))) != 1): 
			coeff_matrix[i][i+((n+1)*(o+1))] = -(2/hx**2 + 2/hy**2 + 2/hz**2) #x + hx
		if (i//((n+1)*(o+1)) > 0): 
			coeff_matrix[i][i-((n+1)*(o+1))] = -(2/hx**2 + 2/hy**2 + 2/hz**2) #x - hx
			
	vect_indep = np.zeros((m+1)*(n+1)*(o+1))
	
	for i in range(len(vect_indep)):
		vect_indep[i] = mesh[i//((n+1)*(o+1))][(i//(n+1)%(o+1))][((i%(n+1)))]/(8.854187817*10**(-12))

	field_nodes = la.solve(coeff_matrix, vect_indep)

	return field_nodes



def pic(m,n,o,a,b,L,N, alpha, kappa, E, mass, tsteps, deltat):

	#initial conditions----------------------------------------------------------
	x = np.zeros((N, tsteps)).T
	y = np.zeros((N, tsteps)).T
	z = np.zeros((N, tsteps)).T
	y[0] = np.random.uniform(a/2 - c/2, a/2 +c/2, N)
	z[0] = np.random.uniform(b/2 - d/2, b/2 +d/2, N)
	vx = np.zeros(N)
	vy = np.zeros(N)
	vz = np.zeros(N)
	modv = np.sqrt(2*E/m)
	for i in range(N):
		theta = np.random.uniform(np.pi/2 - alpha/2 ,np.pi/2 + alpha/2, 1)[0]
		phi = np.random.uniform(-alpha/2 ,alpha/2, 1)[0]
		vx[i]= modv * np.cos(phi)*np.sin(theta) + ((rd.random()*2)-1)*(modv * np.cos(phi)*np.sin(theta))*(kappa/100)
		vy[i]= modv * np.sin(phi)*np.sin(theta) + ((rd.random()*2)-1)*(modv * np.sin(phi)*np.sin(theta))*(kappa/100)
		vz[i]= modv * np.cos(theta) + ((rd.random()*2)-1)*(modv * np.cos(theta))*(kappa/100)
	#----------------------------------------------------------------------------

	for step in range(tsteps):
		mesh = linear_scattering(x[step],y[step],z[step],m,n,o,a,b,L,N)
		field_nodes = solve_field(mesh, m,n,o,a,b,L)

	fig = plt.figure(figsize=(20, 14))
	ax = fig.add_subplot(111, projection='3d')
	generate_enclosure(a,b,c,d,L, ax)
	for i in range(len(x.T)):
    	 ax.plot(x.T[i], y.T[i], z.T[i], marker='o', markersize=5)
	
	plt.show()


def generate_enclosure(a,b,c,d,L, ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0,L)
    ax.set_ylim(-a/2, 3/2*a)
    ax.set_zlim(-b/2, 3/2*b)

    x = np.linspace(0, L, 100)
    y = np.linspace(0, a, 100)
    z = np.linspace(0, b, 100)
    yc = np.linspace(a/2 -c/2, a/2 + c/2, 100)
    zd = np.linspace(b/2 -d/2, b/2 + d/2, 100)

    ax.plot(x, 0*np.ones_like(x), 0*np.ones_like(x), 'k')
    ax.plot(x, a*np.ones_like(x), 0*np.ones_like(x), 'k')
    ax.plot(x, 0*np.ones_like(x), b*np.ones_like(x), 'k')
    ax.plot(x, a*np.ones_like(x), b*np.ones_like(x), 'k')

    ax.plot(0*np.ones_like(x), y, 0*np.ones_like(x), 'k')
    ax.plot(0*np.ones_like(x), y, b*np.ones_like(x), 'k')
    ax.plot(0*np.ones_like(x), 0*np.ones_like(x), z, 'k')
    ax.plot(0*np.ones_like(x), a*np.ones_like(x), z, 'k')

    ax.plot(L*np.ones_like(x), y, 0*np.ones_like(x), 'k')
    ax.plot(L*np.ones_like(x), y, b*np.ones_like(x), 'k')
    ax.plot(L*np.ones_like(x), 0*np.ones_like(x), z, 'k')
    ax.plot(L*np.ones_like(x), a*np.ones_like(x), z, 'k')

    ax.plot(L*np.ones_like(x), yc, (b/2-d/2)*np.ones_like(x), 'k')
    ax.plot(L*np.ones_like(x), yc, (b/2+d/2)*np.ones_like(x), 'k')
    ax.plot(L*np.ones_like(x), (a/2-c/2)*np.ones_like(x), zd, 'k')
    ax.plot(L*np.ones_like(x), (a/2+c/2)*np.ones_like(x), zd, 'k')

    ax.plot(0*np.ones_like(x), yc, (b/2-d/2)*np.ones_like(x), 'k')
    ax.plot(0*np.ones_like(x), yc, (b/2+d/2)*np.ones_like(x), 'k')
    ax.plot(0*np.ones_like(x), (a/2-c/2)*np.ones_like(x), zd, 'k')
    ax.plot(0*np.ones_like(x), (a/2+c/2)*np.ones_like(x), zd, 'k')

    X, Z = np.meshgrid(x, x)
    Z[Z > b] = 1
    Y1 = np.ones_like(X)
    plane1 = ax.plot_surface(X, Y1, Z, alpha=0.2, color='k')

    Y2 = np.zeros_like(X)
    plane2 = ax.plot_surface(X, Y2, Z, alpha=0.2, color='k')

    X, Y = np.meshgrid(x, x)
    Y[Y > a] = 1
    Z1 = np.ones_like(X)
    plane1 = ax.plot_surface(X, Y, Z1, alpha=0.2, color='k')

    Z2 = np.zeros_like(X)
    plane2 = ax.plot_surface(X, Y, Z2, alpha=0.2, color='k')

    Z, Y = np.meshgrid(x, x)
    Y[Y > a/2 + c/2] = a/2 + c/2
    Y[Y < a/2 - c/2] = a/2 - c/2
    Z[Z > b/2 + d/2] = b/2 + d/2
    Z[Z < b/2 - d/2] = b/2 - d/2
    X1 = np.ones_like(X)*L
    plane1 = ax.plot_surface(X1, Y, Z, alpha=0.2, color='k')

    X2 = np.ones_like(X)*0
    plane2 = ax.plot_surface(X2, Y, Z, alpha=0.2, color='k')