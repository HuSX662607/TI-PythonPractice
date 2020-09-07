import numpy as np
import math
import matplotlib.pyplot as plt
import PauliMatrices as pm
import SSH

def FiniteSizeDispersion(v,w):
    H = v*np.kron(SSH.hop1 , pm.sx)\
        + w*(np.kron(SSH.hop2 , (pm.sx - pm.I*pm.sy)/2)+np.kron(np.transpose(SSH.hop2) , (pm.sx + pm.I*pm.sy)/2))
    E,Vec = np.linalg.eig(H)
    E = np.real(E)
    E = E.tolist()
    Vec = Vec.tolist()
    Vec.append(E)
    Vec = np.array(Vec)
    Vec = Vec.T[np.lexsort(Vec)].T 
    E=Vec[-1]
    Vec=Vec[:-1]
    return E,Vec

if __name__=="__main__":
    w = 1
    AllE=[];AllVec=[]
    plt.figure()
    for v in np.arange(0,3,0.01):
        E,Vec = FiniteSizeDispersion(v,w)
        AllE.append(E)
        AllVec.append(Vec)
    VecsAtTop=np.transpose(AllVec[5])
    SymEdge=VecsAtTop[9]
    ASymEdge=VecsAtTop[10]

    plt.subplot(121)
    plt.plot(np.arange(0,3,0.01),AllE)
    plt.xlabel('v')
    plt.ylabel('E')

    plt.subplot(122)
    plt.plot(range(1,21),SymEdge,range(1,21),ASymEdge)
    plt.text(60, .025, r'$v=0.05$')
    plt.xlabel('x')
    plt.ylabel('Wavefunction')

    plt.show()
