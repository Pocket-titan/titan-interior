import numpy as np
import matplotlib.pyplot as plt
from tools import upwardsIntegrator, downwardsIntegrator

n = 10000


#Input planet parameters
R1 = 6371*1e3 #Change this
R2 = 0#More layers
R3 = 0#Even more
G = 6.6743e-11


rho_0 = 5515 #Density in layer 0
rho_1 = 0 #Density in layer 1, etc

#Initialise variables
dr = R1/n
M0 = 0
r0 = 0
g0 = 0


interior = upwardsIntegrator(r0,dr,M0,rho_0,R1,n)

interior2 = downwardsIntegrator(0,dr,rho_0,n,interior)

