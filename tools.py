import numpy as np



def upwardsIntegrator(r0, dr, M0, rho_0,R,n):

    r = r0
    M = M0
    G = G = 6.6743e-11

    interior = np.zeros([n,4])


    for i in range(n):

        r += dr
        M += 4*np.pi*rho_0*(r**2)*dr
        g = G*M/(r**2)

        interior[i,0] = R
        interior[i,1] = M
        interior[i,2] = G

    return interior

def downwardsIntegrator(p0,dr,rho_0,n,interior):

    p = p0
    interiorNew = interior

    for i in range(n):

        p += rho_0*interior[n-1,2]*dr
        interiorNew[i,3] = p

    return interiorNew






