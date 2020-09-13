import numpy as np
import math
import matplotlib.pyplot as plt
import PauliMatrices as pm

Nx=10;Ny=60
HopX = np.eye(Nx,k=-1) #\sum_mx |mx+1><mx|
HopY = np.eye(Ny,k=-1)+np.eye(Ny,k=Ny-1) #\sum_my |my+1><my|
OnSiteX = np.eye(Nx)
OnSiteY = np.eye(Ny)
EdgeOnSiteX_L = np.zeros((Nx,Nx))
EdgeOnSiteX_R = np.zeros((Nx,Nx))
EdgeOnSiteX_L[0,0] = 1;EdgeOnSiteX_R[Nx-1,Nx-1] = 1
K=[-math.pi + 2*n*math.pi/Ny for n in range(0,Ny+1)]

Ux = np.eye(Nx)
Uz = np.eye(2)
Uy = []
for y in range(1,Ny+1):
    wy = []
    for ky in K:
        wy.append(np.exp(pm.I*ky*y))
    Uy.append(wy)
    
U = np.kron(Uy,np.kron(Ux,Uz)) / np.sqrt(Ny)

