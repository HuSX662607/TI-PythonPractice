import numpy as np
import math
import matplotlib.pyplot as plt
import PauliMatrices as pm

N=10

hop1=np.eye(N)
hop2=np.eye(N,k=1) #\sum_m |m><m+1|

K=[-math.pi + 2*n*math.pi/N for n in range(0,N+1)]

def SSHdispersion(v,w,K,Energy1,Energy2,Dx,Dy):

    H = v*np.kron(hop1,pm.sx)\
        + w*(np.kron(hop2,(pm.sx-pm.I*pm.sy)/2)+np.kron(np.transpose(hop2),(pm.sx+pm.I*pm.sy)/2))\
        + w*(np.eye(2*N,k=2*N-1)+np.eye(2*N,k=-2*N+1))

    for n in range(0,N+1): #为了画图时把k=-\pi的点画上，多了一位n=0
        U=[]
        for x in range(1,N+1):
            U.append([np.exp(pm.I*K[n]*x),0])
            U.append([0,np.exp(pm.I*K[n]*x)])
        U=np.array(U)/np.sqrt(N)

        Hk=np.matmul(np.conj(np.transpose(U)),np.matmul(H,U))
        print(Hk)
        
        dx=np.real(Hk[1,0])
        dy=np.imag(Hk[1,0])

        Dx.append(dx)
        Dy.append(dy)

        localE=np.linalg.eigvals(Hk)

        Energy1.append(np.real(localE[0]))
        Energy2.append(np.real(localE[1]))

if __name__=='__main__':
   plt.figure()
   for v,w,i in zip([1,1,1,0.6,0],[0,0.6,1,1,1],[1,2,3,4,5]):
       Energy1=[]
       Energy2=[]
       Dx=[]
       Dy=[]
       SSHdispersion(v,w,K,Energy1,Energy2,Dx,Dy)

       plt.subplot(2,5,i)
       plt.xlabel('k')
       if i==1:
          plt.ylabel('E')
       plt.plot(K,Energy1,'k',K,Energy2,'k')

       plt.subplot(2,5,i+5)
       ax = plt.gca()
       ax.spines['right'].set_color('none')
       ax.spines['top'].set_color('none')
       ax.spines['bottom'].set_position(('data',0))
       ax.spines['left'].set_position(("data",0))
       plt.plot(Dx,Dy)

   plt.show()