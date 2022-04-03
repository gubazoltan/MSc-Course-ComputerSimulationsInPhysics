import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.constants import e, hbar, electron_mass

#length scale for the system
l0 = 0.276 #in nanometer

def iteration(psi0, V_vals, tmax, kk, dx, N, itnum):
    """
    Carries out the time evolution of the wavefunction. The time evolution algorithm is described by the split operator method. 

    Parameters
    ----------
    psi0 : list of complex numbers. Initial wavefunction of the wavepacket. 
    
    V_vals : list of floats. The value of the spatially dependent potential in each space point.
    
    tmax : float. Final time where the iteration stops.
    
    kk : list of floats. The possible momentum values for the system
    .
    dx : float. The infinitesimal difference in the position values.
    
    N : integer. The number of position values in the system.
    
    itnum : integer. The number of iterations in the time evolution.

    Returns
    -------
    psi : list of complex values. The final wavefunction of the wave packet.
    
    sps : list of complex lists. A list that contains the snaposhot of the wavefunction at all the time points.

    """
    dt = tmax / itnum #size of time steps

    #initial wavefunction is the first element in the snapshots
    sps = [psi0]
    
    #lets already evaluate the operators that we will use for the time evolution
    V1 = np.exp(-1.j * dt * V_vals / 2.)
    V2 = np.exp(-1.j * dt * V_vals )
    K = np.exp(-1.j * dt * kk * kk / 2.)
    
    #start the split operator method
    psi = V1 * psi0
    
    #add the snapshot
    sps.append(psi)
    
    #set the time to t=0
    t = 0
    
    #carry out an iteration for each time point expect the last 
    while t < tmax :
        psi_f = fftshift(fft(psi)) #take the fourier transform and centralize it 
        psi_fp = K * psi_f  #apply exponentialized momentum**2 
        psi_p = ifft(ifftshift(psi_fp)) #take inverse fourier transform
        sps.append(psi_p) #append the psi_odd
        
        psi = V2 * psi_p #apply potential in real space
        sps.append(psi) #addend the even psi
        t += dt #increase time
        
    #for the last time step repeat again 
    psi_f = fftshift(fft(psi))
    
    psi_fp = K * psi_f #apply momentum
    
    psi_p = ifft(ifftshift(psi_fp)) #take inverse fourier transform
    sps.append(psi_p) #append it to the snapshots

    psi = V1 * psi_p #apply potential
    
    #add the final wavefunction
    sps.append(psi)

    return psi, sps #return the final psi and all the snapshots

def scattering(E0, DE, N, V0, a, itnum):
    """
    Function that finds the scattering amplitude for a scattering process with the characteristic features described below.
    This function can be used when one is also interested in the snapshots of the wavefunction.

    Parameters
    ----------
    E0 : float. The mean energy of the wavepacket. This energy characterizes the k0 component of the wavepacket which characterizes its propagation.
    
    DE : float. Spread of the energy distribution.
    
    N : integer. The number of position values in the system.

    V0 : float. Height of the potential barrier.
    
    a : float. Total width of the potential barrier.
    
    itnum : integer. The number of iterations in the time evolution.

    Returns
    -------
    xs : list of floats. The possible position points.
    
    sps : list of complex lists. A list that contains the snaposhot of the wavefunction at all the time points.

    fw : list of complex floats. The final wavefunction of the wavepacket.
    
    T : float. The transmission coefficient that corresponds to the scattering process

    """
    #carry out a scattering 
    #E0 is the energy of the incoming wave
    k0 = np.sqrt(2*E0) 
    
    Dk = E0/k0 #Dk is the spread of the packet in momentum space
    Dx = 0.5/Dk #spread in real space
    x_size = max(100, 10*Dx) # linear size of the system 
    
    xs = np.linspace(-x_size, x_size, N) #create the system
    dx = xs[1]-xs[0] #dx steps
    kk = np.linspace(-np.pi/dx,np.pi/dx,N) #possible k values
    s = 0.5*k0/DE #extension of wavefunction in energy
    
    tmax = x_size/k0 #maximal time value
    x0 = -x_size/2 #initial position of the wavepacket
    
    width = a/l0 #the width of the potential should be in units of l0
    V_vals = np.zeros_like(xs) + ( abs(xs) <= width / 2 ) * V0 #create the square potential
    #create the initial wavepacket
    factor = 1/(2*np.pi*s**2)**0.25
    psi0 = np.exp(-(xs-x0)**2/(4.*s**2))*np.exp(1.j*k0*xs) * factor
    
    #carry out the numerov method
    fw,sps = iteration(psi0 = psi0, V_vals = V_vals, tmax = tmax, kk = kk, dx = dx, N = N, itnum = itnum)

    #find the scattering coefficient
    T = sum(np.abs(fw[int(N/2)+1:])**2)*dx
    
    #return the position values, the snapshots, the final wavefunction and the transmission coefficient
    return xs, sps, fw, T

def analytical_curve(E,V0,width):
    """
    Return the analytical curve for a scattering of a plane wave with energy E through a potential barrier of height V0 and width width.

    Parameters
    ----------
    E : float. Energy of the scattering plane wave.
    
    V0 : float. Height of the potential barrier.
    
    width : float. Width of the potential barrier.

    Returns
    -------
    ac : float. The transmission coefficient describing the scattering.

    """
    a = width * 1e-9
    
    if E-V0 >= 0 : #classical regime
        k = np.sqrt(2*electron_mass*e*(E-V0))/ hbar
        
        den = 1 + V0**2 / (4 * E * (E-V0) ) * (np.sin(k*a))**2
        
    else:
        k = np.sqrt(2*electron_mass*e*(V0-E))/ hbar
        
        den = 1 + V0**2 / (4 * E * (V0-E) ) * (np.sinh(k*a))**2
    
    ac = np.real(1/den)
    return ac

def get_acurve_energies(Es, V0, width):    
    """
    Get the analytical transmission coefficient for several different energy values.

    Parameters
    ----------
    Es : list of floats. The possible energy values for which we want to evaluate the scattering.
    
    V0 : float. Height of the potential barrier.
    
    width : float. Width of the potential barrier.

    Returns
    -------
    ac : list of floats. The analytical curve for the transmission coefficient.

    """
    #get the analytical values for the same potential
    ac = []
    
    for E in Es:    
        ac.append(analytical_curve(E = E, V0 = V0, width = width))
        
    ac = np.array(ac)
    
    return ac

def iteration_without_snapshot(psi0, V_vals, tmax, kk, dx, N, itnum):
    """
    Time evolution for the split operator method, but now the snapshots are not saved.

    Parameters
    ----------
    psi0 : list of complex numbers. Initial wavefunction of the wavepacket. 
    
    V_vals : list of floats. The value of the spatially dependent potential in each space point.
    
    tmax : float. Final time where the iteration stops.
    
    kk : list of floats. The possible momentum values for the system
    .
    dx : float. The infinitesimal difference in the position values.
    
    N : integer. The number of position values in the system.
    
    itnum : integer. The number of iterations in the time evolution.

    Returns
    -------
    psi : list of complex values. The final wavefunction of the wave packet.

    """
    dt = tmax / itnum #size of time steps

    
    V1 = np.exp(-1.j * dt * V_vals / 2.)
    V2 = np.exp(-1.j * dt * V_vals )
    K = np.exp(-1.j * dt * kk * kk / 2.)
    
    psi = V1 * psi0
    
    
    t = 0
    
    while t < tmax :
        psi_f = fftshift(fft(psi)) #take the fourier transform and centralize it 
        psi_fp = K * psi_f  #apply exponentialized momentum**2 
        psi_p = ifft(ifftshift(psi_fp)) #take inverse fourier transform
        
        psi = V2 * psi_p #apply potential in real space
        
        t += dt #increase time
        
    #for the last time step repeat again 
    psi_f = fftshift(fft(psi))
    
    psi_fp = K * psi_f #apply momentum
    
    psi_p = ifft(ifftshift(psi_fp)) #take inverse fourier transform

    psi = V1 * psi_p #apply potential

    return psi #return the final psi and all the snapshots

def scattering_without_snapshot(E0, DE, N, V0, a, itnum):
    """
    Function that finds the scattering amplitude for a scattering process with the characteristic features described below.
    This function can be used when one is not interested in the snapshots of the wavefunction.

    Parameters
    ----------
    E0 : float. The mean energy of the wavepacket. This energy characterizes the k0 component of the wavepacket which characterizes its propagation.
    
    DE : float. Spread of the energy distribution.
    
    N : integer. The number of position values in the system.

    V0 : float. Height of the potential barrier.
    
    a : float. Total width of the potential barrier.
    
    itnum : integer. The number of iterations in the time evolution.

    Returns
    -------
    xs : list of floats. The possible position points.

    fw : list of complex floats. The final wavefunction of the wavepacket.
    
    T : float. The transmission coefficient that corresponds to the scattering process

    """
    #carry out a scattering 
    #E0 is the energy of the incoming wave
    k0 = np.sqrt(2*E0) 
    
    Dk = E0/k0 #Dk is the spread of the packet in momentum space
    Dx = 0.5/Dk #spread in real space
    x_size = max(100, 10*Dx) # linear size of the system 
    
    xs = np.linspace(-x_size, x_size, N) #create the system
    dx = xs[1]-xs[0] #dx steps
    kk = np.linspace(-np.pi/dx,np.pi/dx,N) #possible k values
    s = 0.5*k0/DE #extension of wavefunction in energy
    
    tmax = x_size/k0 #maximal time value
    x0 = -x_size/2 #initial position of the wavepacket
    
    width = a/l0 #the width of the potential should be in units of l0
    V_vals = np.zeros_like(xs) + ( abs(xs) <= width / 2 ) * V0 #create the square potential
    #create the initial wavepacket
    factor = 1/(2*np.pi*s**2)**0.25
    psi0 = np.exp(-(xs-x0)**2/(4.*s**2))*np.exp(1.j*k0*xs) * factor
    
    fw = iteration_without_snapshot(psi0 = psi0, V_vals = V_vals, tmax = tmax, kk = kk, dx = dx, N = N, itnum = itnum)

    T = sum(np.abs(fw[int(N/2)+1:])**2)*dx
    
    return xs, fw, T

def multi_scattering(E0s, DE, N, V0, a, itnum):
    """
    Function that carrier out the split operator method for several energy values. 

    Parameters
    ----------
    E0s : list of floats. The possible energy values for which we want to evaluate the scattering.

    DE : float. Spread of the energy distribution.
    
    N : integer. The number of position values in the system.

    V0 : float. Height of the potential barrier.
    
    a : float. Total width of the potential barrier.
    
    itnum : integer. The number of iterations in the time evolution.

    Returns
    -------
    
    Ts: list of loats. Contains the scattering amplitudes for the process.

    """
    #contain the scattering amplitudes here
    Ts = []
    
    #for every energy value in E0s
    for E0 in E0s:
        #carry out the scattering calculation
        xs, fw, T = scattering_without_snapshot(E0 = E0, DE = DE, N = N, V0 = V0, a = a, itnum = itnum)
        Ts.append(T)
        
    return np.array(Ts)