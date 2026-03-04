from scipy.io import loadmat
import bemsolver as bem
import numpy as np
import os
import pickle

import matplotlib as mpl

from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.pyplot as plt

# ==============Plotting settings==================

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "Palatino"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"
mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

#===================================================


def find_flow(t: float, x: np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    # no shear flow
    gamma_dot=-1

    U = np.zeros(3)

    U[0] = 0#-1000 
    U[1] = 0
    U[2] = 0

    # Background vorticity
    W = np.zeros(3)  

    W[0] = 0
    W[1] = 0#-gamma_dot
    W[2] = -gamma_dot

    # Rate of strain tensor
    E = gamma_dot/2*np.array([[0,1,0],
                            [1,0,0],
                            [0,0,0]])
    return U, W, E


r = 3
theta =3
c=3
t= np.linspace(-5,5,400)
Rx = r * np.cos(theta*t)
Ry = r* np.sin(theta*t)
Rz = c*t

L = np.sqrt(r**2*theta**2+c**2) * 10

R = np.column_stack((Rz,Ry,Rx))


flag = bem.SlenderCoordinates(R, np.zeros_like(R), flagellum_length=L, flagellum_radius=0.1)
M = flag.construct_mobility_matrix()

RHS = flag.set_boundary_condition(*find_flow(np.zeros(3),0))

f = np.linalg.solve(M, -RHS)

fq = f.reshape(int(len(f)/3),3)/3

print(np.sum(fq,axis=0))






# print(R)

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ranges = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_range = 10#ranges.max() 

    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.set_title("Helical Flagellum")
ax.set_xlabel(r'$x$ [$\mu$m]')
ax.set_ylabel(r'$y$ [$\mu$m]')
ax.set_zlabel(r'$z$ [$\mu$m]')

# ax.plot(Rx,Ry,Rz, color='black')
ax.plot(flag.r[:,0],flag.r[:,1],flag.r[:,2], color='blue')
ax.quiver(flag.r[:,0],flag.r[:,1],flag.r[:,2], fq[:,0],fq[:,1], fq[:,2], color='r')
# ax.view_init(elev=60)




set_axes_equal(ax)
plt.show()