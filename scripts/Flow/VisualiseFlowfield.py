import bemsolver as bem
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set tick direction globally
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

import os
os.environ["PATH"] += ":/usr/bin"
mpl.rcParams['text.usetex'] = True
mpl.rcParams["font.family"]= "DejaVu Sans"
mpl.rcParams["text.latex.preamble"]+= r"\usepackage{amsmath}"

mpl.rcParams["xtick.labelsize"]=13
mpl.rcParams["ytick.labelsize"]=13
mpl.rcParams["axes.labelsize"]=15
mpl.rcParams["axes.titlesize"]=15
mpl.rcParams["legend.fontsize"]=13

# Background flow
U = np.zeros(3)

U[0] = 1
U[1] = 0
U[2] = 0

# Background vorticity
W = np.zeros(3)

gamma_dot=0



W[0] = 0
W[1] = 0
W[2] = -gamma_dot/2

# Rate of strain tensor
E = gamma_dot/2*np.array([[0,1,0],
                          [1,0,0],
                          [0,0,0]])

path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/elongated-mesh-fine/elongated_spheroid_N=320.mat"

plot_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plots/Flowfield"
plot_image_path = "/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/plot_images/Flowfield"
# path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/sphere_refinement/sphere_mesh_h=2.000000e-01.mat"
mesh= bem.Mesh(path)

Nx=200 + 1
Ny=200 + 1
xlim=3
ylim=3

x = np.linspace(-xlim, xlim, Nx)
y = np.linspace(-ylim, ylim, Ny)

xg, yg =np.meshgrid(x,y)
zg=np.zeros_like(xg)

xg = xg.ravel()
yg = yg.ravel()
zg = zg.ravel()
Ng = np.shape(xg)[0]

points = np.vstack((xg, yg, zg)).T


interaction = bem.FlowStokes(mesh,points)

sys = bem.ResistanceProblem(mesh)

psi, force, torque = sys.solve(U,W,E)


U_field =interaction.calc_vector_field(psi,U, W, E)

vmax=1.2*np.max(np.linalg.norm(U_field, axis =1))
# vmax=4

fig = interaction.plot_vector_field(x, y, U_field, vmax= vmax, quiver_density=8)
fig.set_size_inches(10,8)
ax = fig.axes[0]
ax.set_title(f"Flowfield with $\\mathbf{{u_{{\\infty}}}}= \\mathbf{{\\hat{{x}}}}$ $\\mu$m s$^{{-1}}$", fontsize=20)
fig.savefig(f"{plot_path}/Flowfield_u_x={U[0]}.pdf")
fig.savefig(f"{plot_image_path}/Flowfield_u_x={U[0]}.png",dpi=600)

# plt.show()


