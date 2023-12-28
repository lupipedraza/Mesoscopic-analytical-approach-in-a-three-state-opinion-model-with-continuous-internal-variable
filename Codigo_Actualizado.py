import numpy as np
import matplotlib.pyplot as plt


def corrida(n,dt,T,sigma,h,x0):
    Y=np.zeros((int(T/dt),n))
    X=x0.copy();

    for j in range(int(T/(dt))):
            

            #elijo dos agentes
            permutada= np.random.permutation(range(n))    
            tau=[np.random.normal(X[p],sigma) for p in permutada]

            #Modelo
            for i in range(n):
                if X[i]>tau[i]:
                        X[i]=X[i]-h;
                elif X[i]<tau[i]:
                        X[i]=X[i]+h;
            Y[j]=X
            #grafico
        #mean.append(np.mean(X[0]))
    return (Y)
    
#%%
#Una corrida
n=5000 #agentes
dt=1 #delta tiempo
T=5000 #Tiempo de corrida
sigma=1 #variacion de la opinion
h=10**(-3) #delta espacial
X0=np.linspace(-1/2,1/2,num=n) #Condicion inicial

Y=corrida(n,dt,T,sigma,h,X0)
#Guardar
np.savetxt('h_'+str(h)+'.csv', np.array(Y), delimiter=',')
#Cargar
Y = np.loadtxt('h_'+str(h)+'.csv',delimiter=",", dtype=float)

#%%
#Graficar distribucion de opiniones en un determinado tiempo
tiempo=-1

plt.hist(Y[tiempo],bins=100,range=(-1,1))
plt.xlim([-1,1])
plt.ylim([-0.1,n/3])
plt.xlabel("Opinion", size=16 )
plt.ylabel("Number of agents", size=16 )
plt.show()

#%%

#Histogramas por tiempo

Histo2D=[plt.hist(Y[t],bins=100,range=(-1/2,1/2))[0] for t in range(0,T,100)]

Histo2D=np.array(Histo2D)
#%%
# Graficar evolucion temporal


plt.imshow(Histo2D.T,cmap="Purples",extent=[0,T*h,-1/2,1/2],aspect='auto',vmax=500)
#cax = plt.axes([0.95, 0.1, 0.075, 0.8])
#plt.colorbar(cax=cax)
plt.xlabel('Time',size=16)
plt.ylabel('Opinion',size=16)
#plt.xticks(np.linspace(0,1,10),np.linspace(0,int(T),10))
plt.savefig('densidad_tiempo.pdf')
plt.show()



#%%
#Muchas corridas - promedio
n=1000
dt=1
T=5000
sigma=1
h=10**(-3)
X0=np.linspace(-1/2,1/2,num=n)


Val_medio=[]
for i in range(500):
    Y=corrida(n,dt,T,sigma,h,X0)
    Val_medio.append(np.mean(Y[-1]))
    print(i)
    
#Guardar
np.savetxt('h_'+str(h)+'_ValMedio.csv', np.array(Val_medio), delimiter=',')

#Cargar
Val_medio = np.loadtxt('h_'+str(h)+'_ValMedio.csv',delimiter=",", dtype=float)

plt.hist(Val_medio,bins=15,color='purple',density=True )
plt.axvline(0,color='g')
plt.xlim([-0.02,0.02])
#plt.ylim([-0.1,n/3])
plt.xlabel("Opinion", size=16 )
plt.ylabel("simulations", size=16 )
plt.savefig('Histograma.pdf')
plt.show()





#%%
# Varianza
n=1000
dt=1
T=5000
sigma=1
X0=np.linspace(-1/2,1/2,num=n)

Hs=[10**(-2),5*10**(-3),10**(-3),10**(-4)]#,10**(-5)]

V=[]
S=[]
R=[]
time=10
for h in Hs:
    v=0
    for i in range(time):
        Y=corrida(n,dt,int(T/(h*n)),sigma,h,X0)
        np.savetxt('h_'+str(h)+'i'+str(i)+'.csv', np.array(Y), delimiter=',')
        #Y = np.loadtxt('h_'+str(h)+'.csv',delimiter=",", dtype=float)
        v+=np.var(Y,axis=1)/time
    V.append(v)
    R.append(np.max(Y,axis=1)-np.min(Y,axis=1))
    print(h)
#%%
for h in Hs:
    
    #Y=corrida(n,dt,int(T/(h*n)),sigma,h,X0)
    #np.savetxt('h_'+str(h)+'i'+str(i)+'.csv', np.array(Y), delimiter=',')
    Y = np.loadtxt('h_'+str(h)+'.csv',delimiter=",", dtype=float)
    v=np.var(Y,axis=1)
    V.append(v)
    R.append(np.max(Y,axis=1)-np.min(Y,axis=1))
    print(h)

#%%
# Graficar la evolucion de R y var
from bokeh.palettes import Purples,Greens


colores=Purples[6]

t=np.linspace(0,5,T)
Teo_max=np.var(Y[0])*np.exp(2*np.sqrt(2)/(sigma*np.sqrt(np.pi))*(-1+1/2*np.exp(-1/2))*t)
Teo_min=np.var(Y[0])*np.exp(-2*np.sqrt(2)/(sigma*np.sqrt(np.pi))*t)


plt.plot(np.linspace(0,5,int(T/(Hs[0]*n))),V[0],label='$h=10^{-2}$',linestyle='-',color=colores[0])    
plt.plot(np.linspace(0,5,int(T/(Hs[1]*n))),V[1],label='$h=5*10^{-3}$',linestyle='-',color=colores[1])
plt.plot(np.linspace(0,5,int(T/(Hs[2]*n))),V[2],label='$h=10^{-3}$',linestyle='-',color=colores[2])
plt.plot(np.linspace(0,5,int(T/(Hs[3]*n))),V[3],label='$h=10^{-4}$',linestyle='-',color=colores[3])
#plt.plot(np.linspace(0,T*Hs[3],T),V[3],label='$h=10^{-5}$',linestyle='--')
plt.plot(t,Teo_max, label='Theoretical bounds',linewidth=1.5,color='g',linestyle='--')
plt.plot(t,Teo_min,linewidth=1.5,color='g',linestyle='--')
plt.legend()
plt.yscale('log')
plt.xlabel('Time', size=16)
plt.ylabel('Variance', size=16)
plt.savefig('Varianza_Log.pdf')
plt.show()



#plt.plot(np.linspace(0,T,dt),V)    
    
    