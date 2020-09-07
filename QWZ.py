import numpy as np
import math
import matplotlib.pyplot as plt
import PauliMatrices as pm
import QWZ_parameter as par

u=-1.2;mu_L=0;h_L=0;mu_R=0;h_R=0

def MakeStripeH(w,mu_L,h_L,mu_R,h_R):
    H = np.kron(par.OnSiteY , np.kron(par.HopX , (pm.sz + pm.I*pm.sx)/2)) \
        + np.kron(par.OnSiteY , np.kron(np.transpose(par.HopX) , (pm.sz - pm.I*pm.sx)/2)) \
        + np.kron(par.HopY , np.kron(par.OnSiteX , (pm.sz + pm.I*pm.sy)/2)) \
        + np.kron(np.transpose(par.HopY) , np.kron(par.OnSiteX , (pm.sz-pm.I*pm.sy)/2)) \
        + w * np.kron(par.OnSiteY , np.kron(par.OnSiteX , pm.sz)) \
        + h_L * np.kron(np.eye(par.Ny,k=-2) + np.eye(par.Ny,k=2) + np.eye(par.Ny,k=par.Ny-2) + np.eye(par.Ny,k=-par.Ny+2) \
            ,np.kron(par.EdgeOnSiteX_L , np.eye(2))) \
        + h_R * np.kron(np.eye(par.Ny,k=-2) + np.eye(par.Ny,k=2) + np.eye(par.Ny,k=par.Ny-2) + np.eye(par.Ny,k=-par.Ny+2) \
            ,np.kron(par.EdgeOnSiteX_R , np.eye(2))) \
        + mu_R * np.kron(np.eye(par.Ny) , np.kron(par.EdgeOnSiteX_R , np.eye(2))) \
        + mu_L * np.kron(np.eye(par.Ny) , np.kron(par.EdgeOnSiteX_L , np.eye(2)))
    
    return H

def QWZDispersion(w,mu_L,h_L,mu_R,h_R):
    H = MakeStripeH(w,mu_L,h_L,mu_R,h_R)

    Ux = np.eye(par.Nx)
    Uz = np.eye(2)
    Uy = []
    for y in range(1,par.Ny+1):
        wy = []
        for ky in par.K:
            wy.append(np.exp(pm.I*ky*y))
        Uy.append(wy)
    
    U = np.kron(Uy,np.kron(Ux,Uz))
    
    E = [];Vec = []

    for n in range(0,par.Ny+1):
        Uk = U[:,2*par.Nx*n:2*par.Nx*(n+1)]
        Hk = np.matmul(np.conj(np.transpose(Uk)),np.matmul(H,Uk)) 
        Ek,VecK = np.linalg.eig(Hk)

        Ek = np.real(Ek)
        Ek = Ek.tolist()
        VecK = VecK.tolist()
        VecK.append(Ek)
        VecK = np.array(VecK)
        VecK = VecK.T[np.lexsort(VecK)].T 
        Ek=VecK[-1]
        VecK=VecK[:-1]

        E.append(Ek)
        Vec.append(VecK)

    return E,Vec

if __name__=='__main__':
   E,Vec = QWZDispersion(u,mu_L,h_L,mu_R,h_R)

   plt.figure()
   plt.plot(par.K,E)
   plt.show()
