def demoprint():
    print("This function is used to check version control")

import time
import itertools
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar, k
from numba import jit

@jit(nopython=True)
def matrixexponen(n,a):

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

@jit(nopython=True)
def lindsolve(params):
    deltat, epsilon0, dV, Temp, w, dtnum, tintnum, gammat = params
    f = w/(2*np.pi)
    T = 1/f
    dt = T*dtnum
    tint = T*tintnum
    ts = np.arange(0, tint, dt, dtype=np.float64)
    tsu = np.arange(0,T,dt, dtype = np.float64)

    #find the initial ground and excited states
    x = epsilon0
    d = deltat

    gr = np.array([(-x-np.sqrt(x**2+d**2))/d ,1], dtype = np.complex128) #define the ground state
    gr_n = gr/np.linalg.norm(gr)                                         #make it normalized
    psi_g = gr_n.copy().reshape(-1,1) #make a column vector
    rho_g = psi_g @ psi_g.T #make the density matrix
    psi_g = rho_g.T.copy().reshape(-1,1) #vectorize the density matrix

    #do the same for the excited state
    exc = np.array([(-x+np.sqrt(x**2+d**2))/d ,1], dtype = np.complex128)
    exc_n = exc/np.linalg.norm(exc)
    psi_e = exc_n.copy().reshape(-1,1)
    rho_e = psi_e @ psi_e.T #same here
    psi_e = rho_e.T.copy().reshape(-1,1)

    #now psi_g and psi_e are the vectorized forms of the initial density matrices

    #find the time evolving operators
    unitary = liouvillians(deltat=deltat, epsilon0=epsilon0, dV=dV, Temp=Temp, w=w, gammat=gammat, dt=dt,tsu=tsu)

    #carry out the time evolution
    nRt_g, rhogt = time_evolve(psi0=psi_g, unitary=unitary, ts=ts, tsu = tsu)
    nRt_e, rhoet = time_evolve(psi0=psi_e, unitary=unitary, ts=ts, tsu = tsu)

    sinwt = np.sin(w*ts)
    coswt = np.cos(w*ts)

    #calculate the quantum resistance and the capacitance
    nrt_fourier_r_g =(2/dV)*(1.0/tint)*e*nRt_g*coswt*(-1.0*w)
    Rinv_g = np.trapz(nrt_fourier_r_g, ts, dt)
    Rinv_g = np.real(Rinv_g)

    nrt_fourier_c_g =(2/dV)*(1.0/tint)*e*nRt_g*sinwt
    C_g = np.trapz(nrt_fourier_c_g, ts, dt)
    C_g = np.real(C_g)

    nrt_fourier_r_e =(2/dV)*(1.0/tint)*e*nRt_e*coswt*(-1.0*w)
    Rinv_e = np.trapz(nrt_fourier_r_e, ts, dt)
    Rinv_e = np.real(Rinv_e)

    nrt_fourier_c_e =(2/dV)*(1.0/tint)*e*nRt_e*sinwt
    C_e = np.trapz(nrt_fourier_c_e, ts, dt)
    C_e = np.real(C_e)
    return C_g, C_e, Rinv_g, Rinv_e , nRt_g, nRt_e, rhogt, rhoet

@jit(nopython=True)
def liouvillians(deltat, epsilon0, dV, Temp, w, gammat, dt, tsu):
    unitary = np.zeros((4,4,len(tsu)),dtype=np.complex128)
    for idx,t in enumerate(tsu):
        hcoef=epsilon0-e*dV*np.sin(w*t)
        H=np.array([[0,deltat/2],[deltat/2,hcoef]],dtype=np.complex128)/hbar
        ii=np.eye(2,dtype=np.complex128)
        pre=np.kron(ii,H)
        post=np.kron(H,ii)
        hamil = -1j*(pre - post)
        d  = deltat
        x  = hcoef
        gr = np.array([(-x-np.sqrt(x**2+d**2))/d ,1], dtype = np.complex128)
        exc = np.array([(-x+np.sqrt(x**2+d**2))/d ,1], dtype = np.complex128)
        gr_n = gr/np.linalg.norm(gr)
        exc_n = exc/np.linalg.norm(exc)
        esplit = np.sqrt(d**2+x**2)
        if Temp == 0: #if the temperature is zero then we have to use these rates
            gamma1 = gammat
            gamma2 = 0
        else: #otherwise use these rates
            gamma1 = gammat*(1+1/(np.exp(esplit/(k*Temp))-1))
            gamma2 = gammat/(np.exp(esplit/(k*Temp))-1)

        #calculate the two jump operators from the instantaneiouiuiuoouus ground and excited state
        cop1 = np.array([[gr_n[0]*exc_n[0].conjugate(),gr_n[0]*exc_n[1].conjugate() ],
                      [gr_n[1]*exc_n[0].conjugate(),gr_n[1]*exc_n[1].conjugate()]],dtype = np.complex128)*np.sqrt(gamma1)

        cop2 = np.array([[exc_n[0]*gr_n[0].conjugate(),exc_n[0]*gr_n[1].conjugate()],
                      [exc_n[1]*gr_n[0].conjugate(),exc_n[1]*gr_n[1].conjugate()]], dtype = np.complex128)*np.sqrt(gamma2)

        #and we store the time dependent operators in this list so we can use them to find the expectation value
        #of the energy

        ii=np.eye(2,dtype=np.complex128)
        #carry out the vectorization method for both of them
        copd1=np.conj(cop1.T)
        cdc1=copd1@cop1
        cdct1=cdc1.T
        term11=np.kron(cop1.conj(),cop1)
        term12=np.kron(ii,cdc1)+np.kron(cdct1,ii)

        copd2=np.conj(cop2.T)
        cdc2=copd2@cop2
        cdct2=cdc2.T
        term21=np.kron(cop2.conj(),cop2)
        term22=np.kron(ii,cdc2)+np.kron(cdct2,ii)

        #return the Liouvillian superoperator
        liouvillian=hamil+term11-0.5*term12+term21-0.5*term22
        uu = matrixexponen(4,liouvillian*dt)
        unitary[:,:,idx] = uu

    return unitary

@jit(nopython=True)
def time_evolve(psi0,unitary,ts,tsu):
    #initial state
    psi = psi0
    #collect the expected occupations here
    nRt = np.zeros(len(ts),dtype=np.float64)
    rhot = np.zeros((len(ts),2,2),dtype = np.complex128)

    #occupation operator for the right dot
    nrop      = np.array([[0,0],[0,1]],
              dtype=np.complex128)
    #density matrix for the initial state
    psim = psi.T.reshape((2,2))
    #calculate the expected occupation of the right dot for the initial state
    nr = np.trace(nrop @ psim)
    nr = np.real(nr)
    nRt[0] = nr
    rhot[0,:,:] = psim

    for idx,t in enumerate(ts[:-1]):
        mat=np.ascontiguousarray(unitary[:,:,idx%len(tsu)]) #take the liouvillian at time point t
        psi=mat@psi #make the time evolution (calculate psi(t+dt) )
        psim=psi.T.reshape((2,2))
        rhot[idx+1,:,:] = psim
        nr=np.trace(nrop @ psim) #calculate the expected occupation
        nr=np.real(nr)
        nRt[idx+1] = nr
    return nRt, rhot

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
         "Ce" : None,
         "Rinvg" : None,
         "Rinve" : None,
         "nRtg" : None,
         "nRte" : None,
         "rhotg" : None,
         "rhote" : None
        #"edotg" : None,
        #"edote" : None
        }
