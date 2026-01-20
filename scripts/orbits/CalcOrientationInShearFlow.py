import bemsolver as bem
import numpy as np
from dataclasses import asdict

gamma_dot=0

def find_flow(t,x):
    U = np.zeros(3)

    U[0] = 0#gamma_dot * x[1]
    U[1] = 0
    U[2] = 0

    # Background vorticity
    W = np.zeros(3)  

    W[0] = 0
    W[1] = 0
    W[2] = -gamma_dot

    # Rate of strain tensor
    E = gamma_dot/2*np.array([[0,1,0],
                              [1,0,0],
                              [0,0,0]])
    return U, W, E




path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/elongated-mesh-fine/elongated_spheroid_N=320.mat"


mesh=bem.Mesh(path)


initial_orientation = np.array([0,0,0])
initial_position    = np.array([0,0,0])

sys=bem.MobilityProblem(mesh,
                        initial_position=initial_position,
                        initial_orientation=initial_orientation)
dt=0.01
T=100

solution = sys.RBM_over_time(T,dt,flow_function=find_flow) 
print(solution.omega)

# datapath ="/home/sjoerd-buitjes/University/Master-Thesis/Master-Thesis-Project/Data/BEM/python-BEM/trajectories/Shear-flow"
# np.savez(f"{datapath}/solution_T={T}_dt={dt}_shear={gamma_dot}.npz", **asdict(solution))