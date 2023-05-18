import numpy as np


def create_layers(layers):
    """
    Pass layers in format:
    ```
    [
        [r0, r1, rho_0],
        ...
    ]
    or
    [
        [radius, rho],
        ...
    ]
    or
    [
        [r0, r1, rho_0, K, \\alpha],
        ...
    ]
    or
    [
        [radius, rho, K, \\alpha],
        ...
    ]
    or
        [radius,rho,K,T, \\alpha, cp or dT/dr, conductive or convective] (0 for conductive, 1 for convective)
    ```
    where K = bulk modulus, \\alpha = thermal expansivity
    """
    layers = np.array(layers)

    if layers.shape[1] in [2, 4]:
        new_layers = np.zeros([len(layers), layers.shape[1] + 1])

        for i in range(len(layers)):
            start = 0 if i == 0 else new_layers[i - 1, 1]
            end = start + layers[i, 0]
            assert (start + end) > 0
            new_layers[i, :] = [start, end, *layers[i, 1:]]

        return new_layers

    if layers.shape[1] in [3, 5]:
        for i, layer in enumerate(layers):
            [r0, r1, rho, *rest] = layer
            assert r0 >= 0 and r1 >= 0
            assert r1 > r0

            if i == 0:
                assert r0 == 0

            if i > 0:
                assert r0 == layers[i - 1][1]

        return layers
    
    if layers.shape[1] in [6,7]:
        
        new_layers = np.zeros([len(layers), layers.shape[1] + 1])

        for i in range(len(layers)):
            start = 0 if i == 0 else new_layers[i-1,1]
            end = start + layers[i,0]
            assert (start+end)>0
            new_layers[i, :] = [start, end, *layers[i, 1:]]

        return new_layers

    raise Exception()


def integrate_layers(layers, integrate_density=False, num_steps=1000):
    """
    Layers start from center:
    [
        [0, 500, 1000],
        [500, 6371 * 1e3, 2000],
    ]

    Returns:
    [layer, step, [r, m, g, p]]
    """
    layers = np.array(layers)

    assert layers[0][0] == 0
    values = np.zeros([len(layers), num_steps, 4])

    for i, layer in enumerate(layers):
        [r0, r1, rho_0] = layer[0:3]
        assert r1 > r0

        if i > 0:
            if not r0 == layers[i - 1][1]:
                raise Exception("Layers must be contiguous")

        if i < len(layers) - 1:
            if not r1 == layers[i + 1][0]:
                raise Exception("Layers must be contiguous")

        dr = (r1 - r0) / num_steps
        # if we are starting at the center, take 0, else take the last mass of the last layer
        M = 0 if i == 0 else values[i - 1, -1, 1]
        interior = integrate_upwards(r0, dr, M, rho_0, num_steps)

        values[i, :, :] = interior

    for j, layer in enumerate(reversed(layers)):
        i = len(layers) - j - 1
        [r0, r1, rho_0] = layer[0:3]

        assert r1 > r0
        dr = (r1 - r0) / num_steps
        # if we are starting at the surface, take 0, else take the last pressure of the last layer
        p0 = 0 if j == 0 else values[i + 1, 0, 3]
        ps = integrate_downwards(p0, dr, rho_0, values[i, :, :])

        values[i, :, 3] = ps

    

    return values

def integrate_density(layers,values,num_steps=1000):
    
    converged = False

    iteration_counter = 0
    new_values = np.zeros([len(layers),num_steps,6])

    l = len(layers)

    while converged==False:

        

        #Downward:
        for i, layer in enumerate(reversed(layers)):

            e = len(layers) - i -1
                
            if iteration_counter==0:

                new_values[e,:,:4] = values[e,:,:]
                #new_values[i,:,2] = rho
                if i == 0:
                    new_values[e,0,4] = layer[3]
                elif i > 0:
                    new_values[e,0,4] = new_values[e-1,-1,4]

            dr = (new_values[e,num_steps-1,0] - new_values[e,0,0])/num_steps
            #dp = (new_values[e,num_steps-1,3] - new_values[e,0,3])/num_steps

            if i==0:

                T = layer[4]
                p = 147*1e3
                
                for j in range(0,num_steps):

                    dp = new_values[e,num_steps-j-1,0] - new_values[e,num_steps-1,0]

                    T += layer[6]*dr
                    new_values[e,num_steps-j-1,4] = T
                    new_values[e,num_steps-j-1,5] = layer[2]*(1-layer[5]*(new_values[e,num_steps-j-1,4]-new_values[e,num_steps-1,4]) + (1/layer[3])*dp)
                    p += new_values[i,j,5] * new_values[i,j,2] * dr
                    new_values[i,j,3] = p
                    

            elif i > 0:

                if layer[6] < 0.9:
                    dT = layer[6]
                    T = layer[4]
                    p = new_values[i-1,0,3]

                    for j in range(0,num_steps):

                        dp = new_values[e,num_steps-j-1,0] - new_values[e,num_steps-1,0]

                        T += dT*dr
                        new_values[e,num_steps-j-1,4] = T
                        new_values[e,num_steps-j-1,5] = layer[2]*(1-layer[5]*(new_values[e,num_steps-j-1,4]-new_values[e,num_steps-1,4]) + (1/layer[3])*dp)
                        p += new_values[i,j,5] * new_values[i,j,2] * dr
                        new_values[i,j,3] = p


                elif layer[6] >0.9:

                    T = layer[4]
                    p = new_values[i-1,0,3]

                    dT = layer[5]*new_values[e,num_steps-1,2]*T/layer[6]

                    new_values[e,num_steps-1,4] = new_values[e-1,0,4]
                    
                    for j in range(0,num_steps):

                        dp = new_values[e,num_steps-j-1,0] - new_values[e,num_steps-1,0]
                        dr = new_values[e,num_steps-j-1,0] - new_values[e,num_steps-1,0]

                        new_values[e,num_steps-j-1,4] = layer[4] + dT*dr
                        new_values[e,num_steps-j-1,5] = layer[2]*(1-layer[5]*dT*dr + (1/layer[3])*dp)
                        p += new_values[i,j,5] * new_values[i,j,2] * dr
                        new_values[i,j,3] = p




                '''
                T = new_values[i-1,0,3]

                for j in range(1,num_steps):

                    dT = layer[4]*new_values[i,num_steps-j,2]*new_values[i,num_steps-j+1,3]/layer[5]
                    T += dT*(new_values[i,num_steps-j,0]-new_values[i,(num_steps-j+1),0])
                    new_values[i,num_steps-j,4] = T'''
        
        #Upward

        M = 0
        g=0
        new_values[0,0,1] = M
        new_values[0,0,2] = g
        p = new_values[0,0,5] * new_values[0,1,2] * new_values[0,0,0]

        for i, layer in enumerate(layers):

            dr = -(new_values[i,-1,0]-new_values[i,0,0])/num_steps

            

            for j in range(1,num_steps):
                v = (4/3)*np.pi*new_values[i,j,0]**3 - (4/3)*np.pi*new_values[i,j-1,0]**3
                M += new_values[i,j,5]*v
                new_values[i,j,1] = M
                g = 6.67e-11 * M / new_values[i,j,0]**2
                new_values[i,j,2] = g
                
            
            '''for j in range(1,num_steps):

                rho_new = rho*(1 - layer[6]*(new_values[i,j,5]-new_values[i,j-1,5]) + 1/(layer[3] + layer[4]*(new_values[i,j,3]-new_values[i,j-1,3])))
                new_values[i,j,4] = rho_new'''


        if iteration_counter > 0:

            difference = new_values - previous_values
            rms_density = np.linalg.norm(difference[:,:,4])
            if rms_density < 1:
                converged = True

        iteration_counter += 1

        previous_values = new_values

    return new_values
    



            










def integrate_upwards(r0, dr, M0, rho_0, num_steps=1000):
    """
    Integrate a layer upwards and return [[r, M, g]]
    """
    r = r0
    M = M0
    G = 6.6743e-11

    interior = np.zeros([num_steps, 4])

    for i in range(num_steps):
        r += dr
        M += 4 * np.pi * rho_0 * (r**2) * dr
        g = G * M / (r**2)

        interior[i, 0] = r
        interior[i, 1] = M
        interior[i, 2] = g

    return interior


def integrate_downwards(p0, dr, rho_0, interior):
    """
    Integrate a layer downwards and return an array with the pressures
    """
    num_steps = len(interior)
    ps = np.zeros(num_steps)
    ps[0] = p = p0

    for i in range(num_steps):
        g = interior[num_steps - i - 1, 2]
        p += rho_0 * g * dr
        ps[num_steps - i - 1] = p

    return ps


def compute_mass(layers):
    """
    Add up the mass of layers as spherical shells with constant density
    """
    M = 0

    for layer in layers:
        [r0, r1, rho_0] = layer[0:3]
        M += 4 * np.pi * rho_0 * (r1**3 - r0**3) / 3

    return M


def compute_moment_of_inertia(layers):
    """
    Add up the moment of intertia of layers as spherical shells with constant density
    """
    MoI = 0

    for layer in layers:
        [r0, r1, rho_0] = layer[0:3]
        MoI += 8 * np.pi * rho_0 * (r1**5 - r0**5) / 15

    return MoI
