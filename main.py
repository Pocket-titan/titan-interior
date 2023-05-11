import numpy as np
import matplotlib.pyplot as plt
from tools import upwardsIntegrator, downwardsIntegrator

n = 10000


#Input planet parameters
R1 = 2575*1e3 #Change this
R2 = 0#More layers
R3 = 0#Even more
G = 6.6743e-11


rho_0 = 1880 #Density in layer 0
rho_1 = 0 #Density in layer 1, etc

#Initialise variables
dr = R1/n
M0 = 0
r0 = 0
g0 = 0


interior = upwardsIntegrator(r0,dr,M0,rho_0,n)

interior2 = downwardsIntegrator(0,dr,rho_0,n,interior)


#Verification

massError = interior2[-1,1] - 1345.70*1e20
massErrorPercent = massError/(1345*1e20) * 100
radiusError = interior2[-1,0] - R1
radiusErrorPercent = (radiusError/R1) * 100

print(f'Mass error: {massError} which is a percentage error of {massErrorPercent}%')
print(f'Radius error: {radiusError} which is a percentage error of {radiusErrorPercent}%')



fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(interior2[:,1],interior2[:,0])
ax2.plot(interior2[:,2],interior2[:,0])
ax3.plot(interior2[:,3],interior2[:,0])

plt.show()



