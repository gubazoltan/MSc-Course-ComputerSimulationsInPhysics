import numpy as np
from scipy.constants import e, hbar, k

class Measurement:
    def __init__(self,name):
        self.name = name
        self.setup = {
            "deltat" : None,
            "epsilon0" : None,
            "dV" :None,
            "Temp" : None,
            "w" : None,
            "dtnum": None,
            "tintnum" : None,
            "gammat" : None
        }
        self.ismeasured = False

    def measure(self):
        ress = lindsolve(np.array([self.setup["deltat"],self.setup["epsilon0"],self.setup["dV"],self.setup["Temp"],
        self.setup["w"],self.setup["dtnum"],self.setup["tintnum"],self.setup["gammat"]],dtype = np.float64))
        returnvalue = Result(setupdata = self.setup)
        for idx, key in enumerate(list(returnvalue.datas.keys())):
            returnvalue.datas[key] = ress[idx]
        return returnvalue
    
    
class Result:
    def __init__(self,setupdata,name = None):
        self.name = name
        self.setup = setupdata
        self.datas = {
         "Cg" : None,
         "Gg" : None,
         "Ce" : None,
         "Ge" : None,
         "rhogt" : None,
         "rhoet" : None
        }
        

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

def vected_comm(deltat, hcoef):
    """
    Function that calculates the vectorized form of the commutator in the Lindblad equation.

    Parameters
    ----------
    deltat : float, tunneling matrix element.
    
    hcoef : float, time dependent detuning in the Hamiltonian.

    Returns
    -------
    vected_comm : 4x4 matrix, the vectorized commutator in the Lindblad equation.

    """
    #create the time dependent hamiltonian
    H = np.array([ [0, deltat / 2], 
                   [deltat / 2, hcoef] ], dtype = np.complex128) / hbar 
    
    #now carry out the vectorization of the hamiltonian first
    #2x2 identity
    ii = np.eye(2, dtype=np.complex128) 

    #do vectorization for the commutator of the hamiltonian
    pre = np.kron(ii, H) #first part of the commutator
    post = np.kron(H, ii) #second part of the commutator

    #evaluate the commutator
    vected_comm = -1.j * (pre - post)
    
    #return the vectorized commutator
    return vected_comm

def snapshot_states(deltat,hcoef):
    """
    Function that calculates the instantaneous eigenstates of the time dependent Hamiltonian.

    Parameters
    ----------
    deltat : float, tunneling matrix element.
    
    hcoef : float, time dependent detuning in the Hamiltonian.

    Returns
    -------
    gr_n : array, the instantaneous normalized ground state of the Hamiltonian.
    
    exc_n : array, the instantaneous normalized excited state of the Hamiltonian.

    """
    #shorthand notation for the evaluation of the eigenstates
    d  = deltat  
    x  = hcoef 
    
    #now evaluate the snapshots of the ground and the excited states for the system
    #(plug in numbers into the analytical formula)
    gr = np.array([( - x - np.sqrt( x**2 + d**2 )) / d , 1], dtype = np.complex128)
    exc = np.array([( - x + np.sqrt( x**2 + d**2 )) / d , 1], dtype = np.complex128)
    
    #properly normalize the states
    gr_n = gr/np.linalg.norm(gr)
    exc_n = exc/np.linalg.norm(exc)
    
    #return the normalized snapshots
    return gr_n, exc_n

def relaxation_rates(Temp, gammat, deltat, hcoef):
    """
    Function that finds the temperature dependent downhill and uphill relaxation rates.

    Parameters
    ----------
    Temp : float, temperature of the system.
    
    gammat : float, general relaxation rate.
    
    deltat : float, tunneling matrix element.
    
    hcoef : float, time dependent detuning in the Hamiltonian.

    Returns
    -------
    gamma1 : float, downhill relaxation rate.
    
    gamma2 : float, uphill relaxation rate.

    """
    #energy difference between the ground and excited states
    esplit = np.sqrt(deltat**2+hcoef**2)
    
    #find the uphill and downhill relaxation rates
    if Temp == 0: #if the temperature is zero then we have only downhill relaxation rate
        gamma1 = gammat
        gamma2 = 0
    else: #otherwise use these rates
        gamma1 = gammat * ( 1 + 1 / ( np.exp(esplit / ( k * Temp )) - 1 ))
        gamma2 = gammat / ( np.exp( esplit / ( k * Temp )) - 1 )
    
    #return the downhill and uphill relaxation rates
    return gamma1, gamma2


def jump_operators(Temp, gammat, deltat, hcoef):
    """
    Function that returns the jump operators that define the coupling with the environment.
    

    Parameters
    ----------
    Temp : float, temperature of the system.
    
    gammat : float, general relaxation rate.
    
    deltat : float, tunneling matrix element.
    
    hcoef : float, time dependent detuning in the Hamiltonian.

    Returns
    -------
    jop1 : 2x2 matrix, the jump operator which describes the downhill relaxation: |ground><excited|.
    
    jop2 : 2x2 matrix, the jump operator which describes the uphill relaxation: |excited><ground|.

    """
    
    #calculate the normalized ground and excited states
    gr_n, exc_n = snapshot_states(deltat = deltat, hcoef = hcoef)
    
    #calculate the relaxation rates
    gamma1, gamma2 = relaxation_rates(Temp = Temp, gammat = gammat, deltat = deltat, hcoef = hcoef)
    
    #calculate the two jump operators from the instantaneour ground and excited state
    #downhill process: \ground><excited|
    jop1 = np.array([[ gr_n[0] * exc_n[0].conjugate(), gr_n[0] * exc_n[1].conjugate() ],
                  [ gr_n[1] * exc_n[0].conjugate() , gr_n[1] * exc_n[1].conjugate() ] ], dtype = np.complex128) * np.sqrt(gamma1)

    #uphill process: \excited><ground|
    jop2 = np.array([[ exc_n[0] * gr_n[0].conjugate(), exc_n[0] * gr_n[1].conjugate() ],
                  [ exc_n[1] * gr_n[0].conjugate(), exc_n[1] * gr_n[1].conjugate() ] ], dtype = np.complex128) * np.sqrt(gamma2)
    
    #return the two jump operators
    return jop1, jop2

def vectorize_jumpoperator(jop):
    """
    Function that calculates the vectorized terms in the Lindblad equation which corresponds to one of the jump operators.

    Parameters
    ----------
    jop : 2x2 matrix, jump operator that is to be vectorized.

    Returns
    -------
    term1 : 4x4 matrix, the vectorized first term in the Lindblad equation.
    
    term2 : 4x4 matrix, the vectorized anticommutator in the Lindblad equation.

    """
    #carry out the vectorization for a single jump operator
    
    #define the identity
    ii = np.eye(2, dtype = np.complex128)
    
    #carry out the vectorization method for a jump operator in the lindbladian
    jop_d = np.conj(jop.T) #dagger of the operator
    
    jdj = jop_d @ jop #operator dagger times the operator
    jdj_t = jdj.T #transpose of the previous product
    
    #vectorized form of the first term in the Lindblad equation
    term1 = np.kron(jop.conj(), jop)
    
    #vectorized form of the anticommutator in the Lindblad equation
    term2 = np.kron(ii, jdj) + np.kron(jdj_t, ii)    
    
    #return the two terms
    return term1, term2

def vected_jops(Temp, gammat, deltat, hcoef):
    """
    Function that calculates the jump operators and returns the vectorized form of 

    Parameters
    ----------
    Temp : float, temperature of the system.
    
    gammat : float, general relaxation rate.
    
    deltat : float, tunneling matrix element.
    
    hcoef : float, time dependent detuning in the Hamiltonian.

    Returns
    -------
    term11 : 4x4 matrix, the vectorized first term in the Lindblad equation corresponding to the downhill jump operator.
    
    term12 : 4x4 matrix, the vectorized anticommutator in the Lindblad equation corresponding to the downhill jump operator.
    
    term21 : 4x4 matrix, the vectorized first term in the Lindblad equation corresponding to the uphill jump operator.
    
    term22 : 4x4 matrix, the vectorized anticommutator in the Lindblad equation corresponding to the uphill jump operator.
    

    """
    #vectorize the jump operators in the lindblad equation
    
    #create the jump operators
    jop1, jop2 = jump_operators(Temp = Temp, gammat = gammat, deltat = deltat, hcoef = hcoef)
    
    #carry out the vectorization method for both of them
    term11, term12 = vectorize_jumpoperator(jop = jop1)
    
    term21, term22 = vectorize_jumpoperator(jop = jop2)
    
    #return the vectorized operators
    return term11, term12, term21, term22

def liouvillians(deltat, epsilon0, dV, Temp, w, gammat, dt, tsu):
    """
    Function that returns a list of vectorized liouvillian operators corresponding to each time step
    for a single period of the drive.

    Parameters
    ----------
    deltat : float, tunneling matrix element.
    
    epsilon0 : float, detuning of the right quantum dot.
    
    dV : float, voltage amplitude of the drive.
    
    Temp : float, temperature of the system.
    
    w : float, frequency of the drive.
    
    gammat : float, general relaxation rate.

    dt : float, difference between sequential time steps.
    
    tsu : list of floats, the time points during a single period of the drive- That is, tsu[0] = 0, tsu[-1] = T-dt. 

    Returns
    -------
    unitary : list of 4x4 matrices, the liouvillian operators corresponding to each time step in tsu.

    """
    
    #define a list of 4x4 matrices with complex datatype
    unitary = np.zeros((4,4,len(tsu)),dtype=np.complex128) #create a container for the exponentialized operators
    
    #for each time step in tsu
    for idx,t in enumerate(tsu):
        
        #calculate the time dependent part of the Hamiltonian
        hcoef = epsilon0 - e * dV * np.sin(w * t) 
        
        #calculate the vectorized form of the commutaton in the lindbladian
        hamil = vected_comm(deltat = deltat, hcoef = hcoef)
        
        #calculate the vectorized jump operators in the lindblad equation
        term11, term12, term21, term22 =  vected_jops(Temp = Temp, gammat = gammat, deltat = deltat, hcoef = hcoef)
        
        #now put together the total whole Liouvillian
        #pay attention to the 1/2 coefficients
        liouvillian = hamil + term11 - 0.5 * term12 + term21 - 0.5 * term22
        
        #calculate the exponentialized operator
        uu = matrixexponen(4,liouvillian*dt)
        
        #add the exponentialized operator to the container
        unitary[:,:,idx] = uu

    #return the exponentialized vectorized liouvillians 
    return unitary

def time_evolve(psi0, unitary, ts, tsu):
    """
    Function that carries out the time evolution of the initial psi0 state with the unitary time evolution operators.

    Parameters
    ----------
    psi0 : list of floats, the vectorized density matrix corresponding to the initial state of the system.
    
    unitary : list of 4x4 matrices, the liouvillian operators corresponding to each time step in tsu.
    
    ts : list of floats, the time points during the whole measurement.
        
    tsu : list of floats, the time points during a single period of the drive- That is, tsu[0] = 0, tsu[-1] = T-dt. 

    Returns
    -------
    rhot : list of 2x2 matrices, the time dependent density matrix corresonding to each time point in the ts list.

    """
    #set psi to the initial state
    psi = psi0.copy()
    
    #collect the time dependent density matrix here
    rhot = np.zeros( (len(ts), 2, 2) , dtype = np.complex128)
    
    #reshape psi to get the initial density matrix
    psim = psi.T.reshape((2,2))
    
    #append it to the density matrix collector
    rhot[0,:,:] = psim

    #carry out the time evolution till the final point
    for idx, t in enumerate(ts[ : -1 ]):
        
        #take the liouvillian at time point t
        #make sure that the periodicity is correct
        mat = np.ascontiguousarray(unitary[ : , : , idx % len(tsu) ])
        
        #time evolve with dt (calculate psi(t+dt))
        psi = mat @ psi 
        
        #construct the density matrix of the state and append the new density matrix to the list
        psim = psi.T.reshape( (2, 2) )
        rhot[ idx + 1, : , : ] = psim
    
    #return the time dependent density matrices as 2x2 matrices
    return rhot

def transform_state(psi):
    """
    Function that takes the row vector of the psi state and then creates the corresponding density matrix and 
    the vectorized density matrix.

    Parameters
    ----------
    psi : list of floats, the row vector corresponding to the psi state, for example psi = np.array([1,0]).

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

def find_CG(rho_t, w, ts, dV, tint, dt):
    """
    Function that finds the parameters R and C of the parallel RC circuit equivalent to the double quantum dot.

    Parameters
    ----------
    rho_t : list of 2x2 matrices, the time dependent density matrix corresonding to each time point in the ts list.
    
    w : float, frequency of the drive.
    
    ts : list of floats, the time points during the whole measurement.
    
    dV : float, voltage amplitude of the drive.
    
    tint : integer, the length of the measurement expressed in terms of the period of the drive.
        
    dt : float, difference between sequential time steps.

    Returns
    -------
    C : float, the capacitance of the equivalent parallel RC circuit.
    
    G : float, the conductance of the equivalent parallel RC circuit (1/R).

    """
    
    #we need these to find the inphase and out-of-phase components of the charge response
    sinwt = np.sin(w*ts)
    coswt = np.cos(w*ts)

    #get the time dependent occupation of the right quantum dot
    nRt = rho_t[:,1,1] 
    
    #calculate the out of phase component of the charge response 
    nRt_fourier_G = -2 / dV / tint * e * nRt * coswt * w
    
    #then integrate
    G = np.trapz(nRt_fourier_G, ts, dt)
    
    #take the real part
    G = np.real(G)
    
    #now calcuate the capacitance
    #calculate the inphase component
    nRt_fourier_C = 2 / dV / tint * e * nRt * sinwt
    
    #then integrate
    C = np.trapz(nRt_fourier_C, ts, dt)
    
    #take the real part
    C = np.real(C)

    #return the capacitance and the conductance
    return C, G

def lindsolve(params):
    """
    Function that carries out the simulation of the double quantum dot and calculates the parameters of the equivalent
    parallel RC circuit.

    Parameters
    ----------
    params : list of parameters that describes the characteristics of the measurement. These are deltat, epsilon0, dV,
    Temp, w, dtnum, tintnum and gammat. Out of these, only dtnum was not introduced so far. 
        dtnum : integer, needed to define the infinitesimal time step of the simulation. dt is defined as T/dtnum. 

    Returns
    -------
    Cg : float, capacitance of the ground state.
    
    Gg : float, conductance of the ground state.
    
    Ce : float, capacitance of the excited state.
    
    Ge : float, conductance of the excited state.
    
    rhogt : list of 2x2 matrices, the time dependent density matrix corresonding to the ground state.
    
    rhoet : list of 2x2 matrices, the time dependent density matrix corresonding to the excited state.

    """
    
    #unpack the parameters of the function
    deltat, epsilon0, dV, Temp, w, dtnum, tintnum, gammat = params
    
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
    
    #find the initial ground and excited states
    gr_n, exc_n = snapshot_states(deltat = deltat, hcoef = epsilon0)
    
    #create the vectorized density matrices
    vr_ground = transform_state(gr_n)
    vr_excited = transform_state(exc_n)

    #find the time evolution superoperators
    unitary = liouvillians(deltat = deltat, epsilon0 = epsilon0, dV = dV, Temp = Temp, w = w, gammat = gammat, dt = dt, tsu = tsu)

    #carry out the time evolution
    #for both of the possible initial states
    #rhogt and rhoet are the time dependent density matrices in 2x2 form 
    rhogt = time_evolve(psi0 = vr_ground, unitary = unitary, ts = ts, tsu = tsu)
    rhoet = time_evolve(psi0 = vr_excited, unitary = unitary, ts = ts, tsu = tsu)

    
    #Using the time dependent density matrix any quantity can be calculated
    #Now we calculate the capacitance and the conductance of the setup
    Cg, Gg = find_CG(rho_t = rhogt, w = w, ts = ts, dV = dV, tint = tint, dt = dt)
    Ce, Ge = find_CG(rho_t = rhoet, w = w, ts = ts, dV = dV, tint = tint, dt = dt)
    
    #return the useful quantites
    return Cg, Gg, Ce, Ge, rhogt, rhoet