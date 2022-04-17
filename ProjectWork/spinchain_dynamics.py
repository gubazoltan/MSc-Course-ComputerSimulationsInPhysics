import numpy as np

def matrixexponen(n,a):
    """
    Function that carries out matrix exponentiation. This method is usually a bit faster than scipy's matrix 
    exponentiation. 
    
    Disclaimer: A few years ago I started to use this method which I borrowed from a friend of mine.
    Since then, I could not find the source of the method. 
    Anyway, it works.

    Parameters
    ----------
    n : integer, the dimension of the matrix to be exponentialized. 
    
    a : array of n arrays of length n, the matrix to be exponentialized.

    Returns
    -------
    e : array of n arrays of length n, the exponentialized matrix.

    """
    #this function is used to do matrix exponentialization
    #this method is usually a bit faster than the usual scipy method this is why I swtiched to it

    q = 6
    a_norm = np.linalg.norm ( a, np.Inf )
    ee = ( int ) ( np.log2 ( a_norm ) ) + 1
    s = max ( 0, ee + 1 )
    a = a / ( 2.0 ** s )
    x = a.copy ( )
    c = 0.5
    ee = np.eye ( n, dtype = np.complex64 ) + c * a
    d = np.eye ( n, dtype = np.complex64 ) - c * a
    p = True

    for kk in range ( 2, q + 1 ):
        c = c * float ( q - kk + 1 ) / float ( kk * ( 2 * q - kk + 1 ) )
        x = np.dot ( a, x )
        ee = ee + c * x
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x
        p = not p

    ee = np.linalg.solve ( d, ee )
    ee = np.ascontiguousarray(ee)

    for kk in range ( 0, s ):
        ee = np.dot ( ee, ee )
    return ee

def spin_operators(N):
    """
    Calculates the tensor product spin operators that act globally on the spin chain. 
    Returns a list called spin_ops with the following structure. spin_ops[i,k,:,:] corresponds to an 
    matrix of size 2^N x 2^N which acts as an identity on all the spins expect the i-th spin, on which it
    acts as a sigma_k operator, where k is taken from the set {x,y,z} and sigma_x,y,z corresponds to the pauli 
    x, y and z operators. 
    
    Parameters
    ----------
    N : integer, the number of spins in the chain.

    Returns
    -------
    spin_ops : list of all the global spin operators.

    """
    
    #sigma0
    s0 = np.eye(2, dtype = np.complex128)
    
    #create sigma matrices
    sx = np.matrix([ [0., 1.], 
                     [1., 0.] ], dtype = np.complex128)
    
    sy = np.matrix([ [0., -1.j],
                     [1.j, 0.] ], dtype = np.complex128)
    
    sz = np.matrix([ [1., 0.], 
                     [0., -1.] ], dtype = np.complex128)
    
    #create a container for the spin operators
    #which are of size 2^N x 2^N
    #we need a container for all the spins and for each spin we have 3 different operators.
    spin_ops = np.zeros((N, 3, 2**N, 2**N), dtype = np.complex128)
    
    #create the tensor product operators that act on the spins in the chain
    #for every spin in the chain
    for i in range(N):
        #copy the 2x2 pauli matrices
        Ox = sx.copy()
        Oy = sy.copy()
        Oz = sz.copy()
        
        #take the kronecker product with the identity for every spin
        #which is on the left side of the spin
        for j in range(i):
            Ox = np.kron(s0, Ox)
            Oy = np.kron(s0, Oy)
            Oz = np.kron(s0, Oz)
        
        #and take kronecker product (from the right) for every spin
        #which is on the right of the spin
        for j in range(i+1,N):
            Ox = np.kron(Ox, s0)
            Oy = np.kron(Oy, s0)
            Oz = np.kron(Oz, s0)            
        
        #now add the operators to the container
        spin_ops[i,0,:,:] = Ox
        spin_ops[i,1,:,:] = Oy
        spin_ops[i,2,:,:] = Oz
        
    #return the list of global spin operators
    return spin_ops

def heis_ham_time_ind(N, J, spinops):
    """
    Contructs the time-independent part of the Heisenberg hamiltonian for the spin chain of N heisenberg spins. 

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    J : float, coupling of neighbouring spins in the Heisenberg chain.
    
    spinops : list of all the global spin operators.

    Returns
    -------
    H0 : Hamiltonian of size 2^N x 2^N. The time-independent part of the total hamiltonian of the spin chain.

    """
    #return the time independent part of the hamiltonian
    
    #create the hamiltonian
    H0 = np.zeros((2**N,2**N), dtype = np.complex128)
    
    #iterate trough the spins expect the last one, which has no neighbours to the right
    for i in range(N-1):
        #add interaction energy to the Hamiltonian
        #interaction is the product of the spin operators for the neighbouring spins
        H0 += - J * ( spinops[i,0,:,:] @ spinops[i+1,0,:,:] + 
                      spinops[i,1,:,:] @ spinops[i+1,1,:,:] + 
                      spinops[i,2,:,:] @ spinops[i+1,2,:,:] )
    
    #return the time independent part of the hamiltonian
    return H0

def magn_field(h, w, t):
    """
    Returns the instantaneous value of the magnetic field which is applied to the first spin of the chain.

    Parameters
    ----------
    h : float, amplitude of the applied magnetic field.
    
    w : float, the frequency of the driving magnetic field.
    
    t : float, the time point in which the field is to be evaluated.

    Returns
    -------
    bt : array of floats, the instantaneous value of the magnetic field.

    """
    #evaluate the components of the magnetic field
    #the applied field implemented here is a field which rotates in the x-z plane
    #this could be changed later on to a field which has arbitrarily direction
    
    bx = h * np.sin(w * t)
    by = 0
    bz = h * np.cos(w * t)
    
    #make an array out of the components
    bt = np.array([bx, by, bz], dtype = np.complex128)
    
    #return te vector of the magnetic field
    return bt

def heis_ham_full(H0, spinops, h, w, t):
    """
    Construct the full time-dependent hamiltonian of the system. 

    Parameters
    ----------
    H0 : Hamiltonian of size 2^N x 2^N. The time-independent part of the total hamiltonian of the spin chain.
    
    spinops : list of all the global spin operators.
    
    h : float, amplitude of the applied magnetic field.
    
    w : float, the frequency of the driving magnetic field.
    
    t : float, the time point in which the field is to be evaluated.

    Returns
    -------
    H0 : Hamiltonian of size 2^N x 2^N. The full, time-dependent hamiltonian of the spin chain. 

    """

    #get the value of the magnetic field
    bt = magn_field(h = h, w = w, t = t)
    
    #add interaction with the magnetic field
    #H_magn = - ( bt_x sigma_x + bt_y sigma_y + bt_z sigma_z )
    #and the sigma operators now correspond to the spin operators of the first spin
    H0 += -1 * ( bt[0] * spinops[0,0,:,:] + 
                 bt[1] * spinops[0,1,:,:] + 
                 bt[2] * spinops[0,2,:,:] ) 
    
    #return the full time dependent hamiltonian
    return H0

def vected_comm(N, H):
    """
    Function that calculates the vectorized form of the commutator in the Lindblad equation.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    H : Hamiltonian of size 2^N x 2^N. The full, time-dependent hamiltonian of the spin chain. 

    Returns
    -------
    vected_comm : Hamiltonian of size 4^N x 4^N. The vectorized form of the total time-dependent hamiltonian.

    """
    
    #now carry out the vectorization of the hamiltonian first
    ii = np.eye(2**N, dtype=np.complex128) #2^N x 2^N identity

    #do vectorization for the commutator of the hamiltonian
    pre = np.kron(ii, H) #first part of the commutator
    post = np.kron(H, ii) #second part of the commutator

    #evaluate the commutator
    vected_comm = -1.j * (pre - post)
    
    #return the vectorized commutator
    return vected_comm

def relaxation_rates(Beta, gammat, esplit):
    """
    Calculate the relaxation rates with which the spin on the right is coupled to the environment.

    TASK: FIND MORE MORE PHYSICAL DESCRITION FOR THE RELAXATION OF THE SPIN
        
    Parameters
    ----------
    Beta : float, inverse temperature. 
    
    gammat : float, general relaxation rate.
    
    esplit : float, energy difference between the the ground and excited states of this spin
    question: in what sense? 

    Returns
    -------
    gamma1 : float, downhill relaxation rate.
    
    gamma2 : float, uphill relaxation rate.

    """
    
    #find the uphill and downhill relaxation rates
    if Beta == np.inf: #if the temperature is zero then we have only downhill relaxation rate
        gamma1 = gammat
        gamma2 = 0
    else: #otherwise use these rates
        gamma1 = gammat * (1 + 1 / (np.exp(esplit * Beta) - 1))
        gamma2 = gammat / (np.exp(esplit * Beta) - 1)
        
    #return the relaxation rates
    return gamma1, gamma2

def jump_operators(N, Beta, gammat, esplit):
    """
    Function that returns the jump operators that define the coupling with the environment.

    Disclaimer: The current implementation of this function considers the effect of the environment
    as a large magnetic moment which points in the y direction (away from the chain). 
    Correspondingly the jump operators describe processes in which the spin jumps into the ground or excited
    states of the sigma_y operator.
        
    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    Beta : float, inverse temperature. 
    
    gammat : float, general relaxation rate.
    
    esplit : float, energy difference between the the ground and excited states of this spin

    Returns
    -------
    sp_y_z : 2^N x 2^N matrix, the operator that describes the downhill relaxation.
    
    sm_y_z : 2^N x 2^N matrix, the operator that describes the uphill relaxation.

    """
    
    #get the relaxation rates
    gamma1, gamma2 = relaxation_rates(Beta = Beta, gammat = gammat, esplit = esplit)
    
    #sigma0
    s0 = np.eye(2, dtype = np.complex128)
    
    #create sigma_y matrix
    sy = np.matrix([ [0., -1.j],
                     [1.j, 0. ] ], dtype = np.complex128)
    
    #get the eigenvectors of the sigma_y operator as they are needed to construct the jump operators
    vals, vecs = np.linalg.eig(sy)
    
    #construct the ladder operators for the sigma_y operator in the eigenbasis of sigma_y
    #S+ in the y eigenbasis looks like this
    sp_y = np.array([[0.,1.],
                     [0.,0.]], dtype = np.complex128)

    #S- in the y eigenbasis looks like this
    sm_y = np.array([[0.,0.],
                     [1.,0.]], dtype = np.complex128)
    
    #now transform this matrix into the eigenbasis of sigma_z.
    #so need to do a basis transfomation
    
    #this is the uphill jump operator
    sp_y_z = np.linalg.inv(vecs) @ (sp_y @ vecs) * np.sqrt(gamma2)
    
    #this is the downhill operator
    sm_y_z = np.linalg.inv(vecs) @ (sm_y @ vecs) * np.sqrt(gamma1)
    
    #these jump operators are only local, we need to construct the global spin operators
    
    #for each spin to the left of the final spin on the right 
    for i in range(N-1):
        #add an identity to the operator from the left
        sp_y_z = np.kron(s0, sp_y_z)
        sm_y_z = np.kron(s0, sm_y_z)
    
    #return the global jump operators
    return sm_y_z, sp_y_z

def vectorize_jumpoperator(N, jop):
    """
    Function that calculates the vectorized terms in the Lindblad equation which corresponds 
    to one of the jump operators.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    jop : matrix of size 2^N x 2^N, the global jump operator to be vectorized

    Returns
    -------
    term1 : 4^N x 4^N matrix, the vectorized first term in the Lindblad equation.
    
    term2 : 4^N x 4^N matrix, the vectorized anticommutator in the Lindblad equation.

    """
    #carry out the vectorization for a single jump operator
    
    ii = np.eye(2**N, dtype = np.complex128)
    
    #carry out the vectorization method for a jump operator in the lindbladian
    jop_d = np.conj(jop.T) #dagger of the operator
    
    jdj = jop_d @ jop #operator dagger times the operator
    jdj_t = jdj.T #transpose of the previous product
    
    #first vectorized
    term1 = np.kron(jop.conj(), jop)
    
    #second vectorized
    term2 = np.kron(ii, jdj) + np.kron(jdj_t, ii)    
    
    #return the two terms
    return term1, term2

def vected_jops(N, Beta, gammat, esplit):
    """
    Function that calculates the jump operators and returns their vectorized form.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    Beta : float, inverse temperature. 
    
    gammat : float, general relaxation rate.
    
    esplit : float, energy difference between the the ground and excited states of this spin.

    Returns
    -------
    term11 : 4^N x 4^N matrix, the vectorized first term in the Lindblad equation corresponding to the downhill jump operator.
    
    term12 : 4^N x 4^N matrix, the vectorized anticommutator in the Lindblad equation corresponding to the downhill jump operator.
    
    term21 : 4^N x 4^N matrix, the vectorized first term in the Lindblad equation corresponding to the uphill jump operator.
    
    term22 : 4^N x 4^N matrix, the vectorized anticommutator in the Lindblad equation corresponding to the uphill jump operator.

    """
    #vectorize the jump operators in the lindblad equation
    
    #create the jump operators
    jop1, jop2 = jump_operators(N = N, Beta = Beta, gammat = gammat, esplit = esplit)
    
    #carry out the vectorization method for both of them
    term11, term12 = vectorize_jumpoperator(N = N, jop = jop1)
    
    term21, term22 = vectorize_jumpoperator(N = N, jop = jop2)
    
    #return the vectorized operators
    return term11, term12, term21, term22

def liouvillians(N, J, h, gammat, esplit, w, Beta, dt, tsu):
    """
    Function that returns a list of vectorized liouvillian operators corresponding to each time step
    for a single period of the drive.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    J : float, coupling of neighbouring spins in the Heisenberg chain.
    
    h : float, amplitude of the applied magnetic field.
    
    gammat : float, general relaxation rate.
    
    esplit : float, energy difference between the the ground and excited states of this spin.
    
    w : float, the frequency of the driving magnetic field.
    
    Beta : float, inverse temperature. 
    
    dt :  float, difference between sequential time steps.
    
    tsu : list of floats, the time points during a single period of the drive- That is, tsu[0] = 0, tsu[-1] = T-dt. 

    Returns
    -------
    unitary : list of 4^N x 4^N matrices, the liouvillian operators corresponding to each time step in tsu.

    """
    #create a container for the exponentialized operators
    unitary = np.zeros(( 4**N, 4**N, len(tsu) ), dtype=np.complex128)
    
    #create spinoperators
    spinops = spin_operators(N = N)
    
    #create the time independent hamiltonian 
    H0 = heis_ham_time_ind(N = N, J = J, spinops = spinops)
    
    for idx,t in enumerate(tsu):
        
        #create the time dependent hamiltonian
        Ht = heis_ham_full(H0 = H0, spinops = spinops, h = h, w = w, t = t)
        
        #calculate the vectorized form of the commutaton in the lindbladian
        hamil = vected_comm(N = N, H = Ht)
        
        #calculate the vectorized jump operators in the lindblad equation
        term11, term12, term21, term22 =  vected_jops(N = N, Beta = Beta, gammat = gammat, esplit = esplit)
        
        #now put together the total whole Liouvillian
        #pay attention to the 1/2 coefficients
        liouvillian = hamil + term11 - 0.5 * term12 + term21 - 0.5 * term22
        
        #calculate the exponentialized operator
        uu = matrixexponen(4**N,liouvillian*dt)
        
        #add the exponentialized operator to the container
        unitary[:,:,idx] = uu
        
    #return the exponentialized vectorized operators
    return unitary

def time_evolve(N, psi0, unitary, ts, tsu):
    """
    Function that carries out the time evolution of the initial psi0 state with the unitary time evolution operators.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    psi0 : list of floats, the vectorized density matrix corresponding to the initial state of the system.
    
    unitary : list of 4^N x 4^N matrices, the liouvillian operators corresponding to each time step in tsu.
    
    ts : list of floats, the time points during the whole measurement.
    
    tsu : list of floats, the time points during a single period of the drive- That is, tsu[0] = 0, tsu[-1] = T-dt.

    Returns
    -------
    rhot : list of 2^N x 2^N matrices, the time dependent density matrix corresonding to each time point in the ts list.

    """
    #set psi to the initial state
    psi = psi0.copy()
    
    #collect the time dependent density matrix here
    rhot = np.zeros( (len(ts), 2**N, 2**N) , dtype = np.complex128)
    
    #density matrix for the initial state
    psim = psi.T.reshape((2**N,2**N))
    
    #append it to the density matrix collector
    rhot[0,:,:] = psim

    #carry out the time evolution till the final point
    for idx, t in enumerate(ts[ : -1 ]):
        mat = np.ascontiguousarray(unitary[ : , : , idx%len(tsu) ]) #take the liouvillian at time point t
        
        psi = mat @ psi #time evolve with dt (calculate psi(t+dt))
        
        #construct the density matrix of the state and append the new density matrix to the list
        psim=psi.T.reshape((2**N,2**N))
        rhot[idx+1,:,:] = psim
    
    #return the time dependent density matrices as 2x2 matrices
    return rhot

def transform_state(psi):
    """
    Function that takes the row vector of the psi state and then creates the corresponding density matrix and 
    the vectorized density matrix.

    Parameters
    ----------
    psi : list of floats, the row vector corresponding to the psi state.

    Returns
    -------
    vected_rho : list of floats, the vectorized form of the density matrix corresponding to state psi.

    """
    #create the density matrix out of the state psi and then create the vectorized density matrix
    #psi is expected to be a usual numpy array
    
    #make a column vector
    #i.e. create the ket vector
    psi_ket = psi.copy().reshape(-1,1)
    
    #make the density matrix
    #take |psi><psi| 
    #<psi| is the bra vector which is the transposed complex conjugated
    rho = psi_ket @ np.conj(psi_ket.T)
    
    #vectorize the density matrix
    vected_rho = rho.T.reshape(-1,1)
    
    #return the vectorized density matrix which corresponds to the psi vector
    return vected_rho

def lindsolve(params):
    """
    Function that carries out the simulation of the spin chain driven by a magnetic field and coupled
    to the environment.

    TASK: MAKE THE FUNCTION ABLE TO TAKE THE INITIAL STATE OF THE SPINS AS A PARAMETER.
    
    Parameters
    ----------
    params : list of parameters that describes the characteristics of the measurement. These are N, J, h, gammat,
    esplit, w, Beta, dtnum, tintnum. Out of these, only dtnum was not introduced so far. 
        dtnum : integer, needed to define the infinitesimal time step of the simulation. dt is defined as T/dtnum. 

    Returns
    -------
    rhot : list of 2^N x 2^N matrices, the time dependent density matrix obtained during the simulation.

    """
    #unpack the parameters of the function
    N, J, h, gammat, esplit, w, Beta, dtnum, tintnum = params
    
    #frequency of the drive
    f = w/(2*np.pi)
    
    #length of one period of the drive
    T = 1/f
    
    #infinitesimal timestep
    dt = T/dtnum
    
    #total time of the time evolution
    tint = T*tintnum

    #time points of the whole time evolution
    ts = np.arange(0, tint, dt, dtype = np.float64)
    
    #time points in a single period
    tsu = np.arange(0, T, dt, dtype = np.float64)
    
    #the initial state is now specified by all spins pointing upwards
    #create an initial state for the system
    psi_up = np.array([1,0])
    psi = np.array([1,0])
    #create the initial psi vector
    for i in range(N-1):
        psi = np.kron(psi, psi_up)
    
    #create the vectorized density matrices
    vr_initial = transform_state(psi)
    
    #find the time evolution superoperators
    unitary = liouvillians(N = N, J = J, h = h, gammat = gammat, esplit = esplit, w = w, Beta = Beta, dt = dt, tsu = tsu)

    #carry out the time evolution
    rhot = time_evolve(N = N, psi0 = vr_initial, unitary = unitary, ts = ts, tsu = tsu)

    #return the time dependent density matrix
    return rhot

def magnetization(N, which, direction, rhot):
    """
    Function that calculates the local magnetization of the sample as a function of time.

    Parameters
    ----------
    N : integer, the number of spins in the chain.
    
    which : integer, the index of the spin for which the magnetization is to be determined. 
    
    direction : string, the direction in which the magnetization is to be determined. 
    Possible values are 'x'. 'y' and 'z'.
    
    rhot : list of 2^N x 2^N matrices, the time dependent density matrix obtained during the simulation.

    Returns
    -------
    magn : list of floats, the time-dependent local magnetization of the sample. 

    """
    
    #sigma0
    s0 = np.eye(2, dtype = np.complex128)
    
    #create sigma matrices
    sx = np.matrix([ [0., 1.], 
                     [1., 0.] ], dtype = np.complex128)
    
    sy = np.matrix([ [0., -1.j],
                     [1.j, 0.] ], dtype = np.complex128)
    
    sz = np.matrix([ [1., 0.], 
                     [0., -1.] ], dtype = np.complex128)
    
    #create local magnetism operator
    if direction == 'x': 
        local_magn = sx
    elif direction == 'y':
        local_magn = sy
    elif direction == 'z':
        local_magn = sz

    #take the kronecker product with the identity for every spin
    #which is on the left side of the spin
    for j in range(which):
        local_magn = np.kron(s0, local_magn)

    #and take kronecker product (from the right) for every spin
    #which is on the right of the spin
    for j in range(which+1,N):
        local_magn = np.kron(local_magn, s0)
        
    #now the local_magn object is an operator which acts on the whole chain and measures the direction
    #of the spin which is the "which"-th in the chain
    #i.e. if which = 1, then local_magn measures the direction of the spin indexed by 1. 
    
    #contain the magnetization
    magn = np.zeros(len(rhot), dtype = np.float64)
    
    for idx,rho in enumerate(rhot):
        magn[idx] = np.real(np.trace(rho @ local_magn))
    
    #now return the magnetization
    return magn