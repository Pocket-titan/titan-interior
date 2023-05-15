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
        [radius,rho,K0,K',T, \\alpha]
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
    
    if layers.shape[1] in [6]:
        
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

    #In doing this, I have treated K as linearly dependent on p (K= K0 + K' * dp, where K' is the derivative w.r.t. pressure, 
    # as provided in Fortes 2012)
    
    converged = False

    

    iteration_counter = 0
    new_values = np.zeros([len(layers),num_steps,6])

    while converged==False:

        

        #Downward:
        for i, layer in enumerate(layers):

            [r0,r1,rho] = layer[0:3]
            assert r1 > r0

            if i > 0:
                if not r0 == layers[i - 1][1]:
                    raise Exception("Layers must be contiguous")

            if i < len(layers) - 1:
                if not r1 == layers[i + 1][0]:
                    raise Exception("Layers must be contiguous")
                
            if iteration_counter==0:

                new_values[i,:,:4] = values[i,:,:]
                new_values[i,:,4] = rho
                if i == 0:
                    new_values[i,0,5] = layer[5]
                elif i > 0:
                    new_values[i,0,5] = new_values[i-1,-1,5]
                
            for j in range(1,num_steps):

                dT = -layer[6]*(new_values[i,j,4]/rho -1 - 1/(layer[3] + layer[4]*(new_values[i,j,3]-new_values[i,j-1,3])))
                T = new_values[i,j-1,5] + dT
                new_values[i,j,5] = T
        
        #Upward

        for i, layer in enumerate(layers):

            [r0,r1,rho] = layer[0:3]
            
            for j in range(1,num_steps):

                rho_new = rho*(1 - layer[6]*(new_values[i,j,5]-new_values[i,j-1,5]) + 1/(layer[3] + layer[4]*(new_values[i,j,3]-new_values[i,j-1,3])))
                new_values[i,j,4] = rho_new


        if iteration_counter > 0:

            difference = new_values - previous_values
            rms_density = np.linalg.norm(difference[:,:,4])
            if rms_density < 1e2:
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
        [r0, r1, rho_0] = layer
        M += 4 * np.pi * rho_0 * (r1**3 - r0**3) / 3

    return M


def compute_moment_of_inertia(layers):
    """
    Add up the moment of intertia of layers as spherical shells with constant density
    """
    MoI = 0

    for layer in layers:
        [r0, r1, rho_0] = layer
        MoI += 8 * np.pi * rho_0 * (r1**5 - r0**5) / 15

    return MoI
