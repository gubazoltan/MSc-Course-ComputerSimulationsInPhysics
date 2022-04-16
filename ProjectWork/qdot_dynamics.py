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
    #this function is used to do matrix exponentialization
    #this method is usually a bit faster than the usual scipy method this is why I swtiched to it

    q = 6
    a_norm = np.linalg.norm ( a, np.Inf )
    ee = ( int ) ( np.log2 ( a_norm ) ) + 1
    s = max ( 0, ee + 1 )
    a = a / ( 2.0 ** s )
    x = a.copy ( )
    c = 0.5
    e = np.eye ( n, dtype = np.complex64 ) + c * a
    d = np.eye ( n, dtype = np.complex64 ) - c * a
    p = True

    for k in range ( 2, q + 1 ):
        c = c * float ( q - k + 1 ) / float ( k * ( 2 * q - k + 1 ) )
        x = np.dot ( a, x )
        e = e + c * x
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x
        p = not p

    e = np.linalg.solve ( d, e )
    e = np.ascontiguousarray(e)

    for k in range ( 0, s ):
        e = np.dot ( e, e )
    return e

def vected_comm(deltat, hcoef):
    H = np.array([ [0, deltat/2], 
                   [deltat/2, hcoef] ], dtype=np.complex128) / hbar #create the time dependent hamiltonian
    
    #now carry out the vectorization of the hamiltonian first
    ii = np.eye(2, dtype=np.complex128) #2x2 identity

    #do vectorization for the commutator of the hamiltonian
    pre = np.kron(ii,H)
    post = np.kron(H,ii)

    #evaluate the commutator
    vected_comm = -1j*(pre - post)
    
    return vected_comm

def snapshot_states(deltat,hcoef):
    #now evaluate the snapshot ground and excited state of the time dependent hamiltonian
    d  = deltat #shorthand notation 
    x  = hcoef #same
    
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
    #energy difference between the ground and excited states
    esplit = np.sqrt(deltat**2+hcoef**2)
    
    #find the uphill and downhill relaxation rates
    if Temp == 0: #if the temperature is zero then we have only downhill relaxation rate
        gamma1 = gammat
        gamma2 = 0
    else: #otherwise use these rates
        gamma1 = gammat * ( 1 + 1 / ( np.exp(esplit / ( k * Temp )) - 1 ))
        gamma2 = gammat / ( np.exp( esplit / ( k * Temp )) - 1 )
        
    return gamma1, gamma2


def jump_operators(Temp, gammat, deltat, hcoef):
    
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
    #carry out the vectorization for a single jump operator
    
    ii = np.eye(2, dtype = np.complex128)
    
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

def vected_jops(Temp, gammat, deltat, hcoef):
    #vectorize the jump operators in the lindblad equation
    
    #create the jump operators
    jop1, jop2 = jump_operators(Temp = Temp, gammat = gammat, deltat = deltat, hcoef = hcoef)
    
    #carry out the vectorization method for both of them
    term11, term12 = vectorize_jumpoperator(jop = jop1)
    
    term21, term22 = vectorize_jumpoperator(jop = jop2)
    
    #return the vectorized operators
    return term11, term12, term21, term22

def liouvillians(deltat, epsilon0, dV, Temp, w, gammat, dt, tsu):
    
    unitary = np.zeros((4,4,len(tsu)),dtype=np.complex128) #create a container for the exponentialized operators
    
    for idx,t in enumerate(tsu):
        
        hcoef = epsilon0 - e * dV * np.sin(w * t) #time dependent coefficient in the hamiltonian
        
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

    return unitary

def time_evolve(psi0, unitary, ts, tsu):
    #set psi to the initial state
    psi = psi0.copy()
    
    #collect the time dependent density matrix here
    rhot = np.zeros( (len(ts), 2, 2) , dtype = np.complex128)
    
    #density matrix for the initial state
    psim = psi.T.reshape((2,2))
    
    #append it to the density matrix collector
    rhot[0,:,:] = psim

    #carry out the time evolution till the final point
    for idx, t in enumerate(ts[ : -1 ]):
        mat = np.ascontiguousarray(unitary[ : , : , idx%len(tsu) ]) #take the liouvillian at time point t
        
        psi = mat @ psi #time evolve with dt (calculate psi(t+dt))
        
        #construct the density matrix of the state and append the new density matrix to the list
        psim=psi.T.reshape((2,2))
        rhot[idx+1,:,:] = psim
    
    #return the time dependent density matrices as 2x2 matrices
    return rhot

def transform_state(psi):
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
    sinwt = np.sin(w*ts)
    coswt = np.cos(w*ts)

    #occupation of the right quantum dot
    nRt = rho_t[:,1,1] 
    
    #calculate the fourier transform of the time dependent occupation
    nRt_fourier_G = -2 / dV / tint * e * nRt * coswt * w
    
    #then calculate the conductance
    G = np.trapz(nRt_fourier_G, ts, dt)
    
    #take the real part
    G = np.real(G)
    
    #now calcuate the capacitance
    #calculate the inphase fourier coefficient
    nRt_fourier_C = 2 / dV / tint * e * nRt * sinwt
    
    #then calculate the capaticance
    C = np.trapz(nRt_fourier_C, ts, dt)
    
    #take the real part
    C = np.real(C)

    #return the capacitance and the conductance
    return C, G

def lindsolve(params):
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