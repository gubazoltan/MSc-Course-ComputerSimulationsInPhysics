import numpy as np
from scipy.constants import hbar, e, electron_mass
import matplotlib.pyplot as plt

#length scale for the system
l0 = 0.276

def square_potential(width, V0, x):
    """
    Returns the value of the square potential at position x. 

    Parameters
    ----------
    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].
    
    V0 : float, height of the square potential. The value of the potential is V0 where it is nonzero.
    
    x : float, the position value at which the potential is evaluated

    Returns
    -------
    V(x) : the value of the potential evaluated at position x.

    """
    #find the value of the potential V(x) for the square potential
    if -width / 2 <= x and x <= width  / 2: #if x is within the region of the potential
        return V0 #then we return the potential
    else: #otherwise the potential is zero
        return 0
    
def gaussian(x, width):
    """
    Returns the value of a gaussian curve evaluated at x. The mean of the curve is 0, the variance of the distribution is width/2

    Parameters
    ----------
    x : float, the position at which the gaussian is evaluated.
    
    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    Returns
    -------
    value : value of the gaussian pdf evaluated at position x.

    """
    #return gaussian curve 
    a = width / 2
    return np.exp( -1 * x**2 / ( 2 * a**2 ) )

def gaussian_potential(width, V0, x):
    """
    Returns the value of the potential V(x) where the potential is a gaussian curve.

    Parameters
    ----------
    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0 : float, height of the gaussian potential. The maximal value of the potential is V0.

    x : float, the position at which the gaussian is evaluated.

    Returns
    -------
    V(x) : the value of the potential evaluated at position x.


    """
    #if we are within the range of the potential then return the corresponding point from the gaussian, otherwise return 0
    if -width / 2 <= x and x <= width  / 2: 
        return gaussian(x = x, width = width)*V0 #scale the gaussian with V0 amplitude
    else: #otherwise the potential is zero
        return 0
    
def k_func(x, potential, width, V0, E):
    """
    Returns the parameter k of the numerov method. 

    Parameters
    ----------
    x : float, the position at which the gaussian is evaluated.

    potential : function, the potential that is present in the system.
    
    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0 : float, height of the potential.

    E : float, energy of the particle that scatters from the potential.

    Returns
    -------
    k : float, parameter of the numerov method evaluated at position x.

    """
    #return the value of k, which is a parameter in the numerov method
    return 2*(E - potential(width = width, V0 = V0, x = x))

def get_ks(xs, dx, potential, width, V0, E):
    """
    Returns the list of the parameters used for the numerov method. If the length of the list xs containing the possible position values is N, then the
    length of the ks list is N+2. 

    Parameters
    ----------
    xs : list of floats, the position values at which the numerov method is used.
    
    dx : float, infinitesimal difference in the position value.
        
    potential : function, the potential that is present in the system.
    
    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0 : float, height of the potential.

    E : float, energy of the particle that scatters from the potential.

    Returns
    -------
    ks : list of floats, the list of parameters of the numerov method. 
    """
    #calculate all the possible k values for the system
    #number of k points is x_num + 3 because we have a k value for each positions but we have also a k for both of the initial psi states that we know (at width/2 and 
    #at width/2+dx)
    
    #collect k values here
    ks = []
    
    #add the k values for the two initial states
    ks.append(k_func(x = xs[-1]+2.*dx, potential = potential, width = width, V0 = V0, E = E)) #width/2+dx
    ks.append(k_func(x = xs[-1]+dx, potential = potential, width = width, V0 = V0, E = E)) #width/2
    
    #now iterate through the position values backwards so that ks contain the k values for decreasing position values
    for x in xs[::-1]:
        #add the k value to the list
        ks.append(k_func(x = x, potential = potential, width = width, V0 = V0, E = E))
    
    #make a numpy array out of this 
    ks = np.array(ks)
    
    #return the ks
    return ks

def iteration(psis, ks, dx):
    """
    Carries out one iteration of the numerov method. During the numerov method, the last two items in the psis list is used
    to calculate the next value which is then added to the list.

    Parameters
    ----------
    psis : list of complex values. List containing the wavefunction of the particle evaluated at the position values. 
    
    ks : list of floats, the list of parameters of the numerov method. 

    dx : float, infinitesimal difference in the position value.

    Returns
    -------
    psis : list of complex values. List containing the wavefunction of the particle evaluated at the position values. 

    """
    #carry out one iteration of the numerov method
    
    #obtain the current state of the method
    n = len(psis) #current number of states that we already obtained
    
    #obtain the k parameters for the 3 states
    kn = ks[n-1]
    kn_p = ks[n-2]
    kn_m = ks[n]
    
    #calculate the new state using the numerov method
    lhs = 2*(1 - 5*dx*dx*kn/(12))*psis[n-1] - (1 + dx*dx*kn_p/(12))*psis[n-2]
    psi_np = lhs /  ((1 + dx*dx*kn_m/(12)))

    #add the new psi state to the list and then return the list
    psis.append(psi_np)
    return psis

def q_func(E):
    """
    Calculate the wavenumber of the free plane wave with energy E.

    Parameters
    ----------
    E : float, energy of the propagating particle.

    Returns
    -------
    q : float, wavenumber of the particle.

    """
    #get the value of the wavevector for the initial state
    return np.sqrt(2*E)

def numerov(psis, ks, dx, x_num):
    """
    Carries out all the iterations of the numerov method for the system. That is, it returns wavefunction evaluated at the position values
    at which the ks are evaluated. The length of this list is x_num+3.

    Parameters
    ----------
    psis : list of complex values. List containing the wavefunction of the particle evaluated at the position values. 

    ks : list of floats, the list of parameters of the numerov method. 

    dx : float, infinitesimal difference in the position value.

    x_num : integer, the number position values which are in the range (-width/2, width/2).

    Returns
    -------
    psis : list of complex values, the wavefunction of the particle evaluated at xs + [width/2, width/2+dx] (at x_num+3 points).

    """
    #do the whole numerov method and obtain the wavefunction
    for i in range(x_num+1):
        psis = iteration(psis = psis, ks = ks, dx = dx)
    return np.array(psis)[::-1]

def findAB(E, dx, psi0, psi_m):
    """
    Returns the complex amplitudes of the leftwards and rightwards propagating parts of the wavefunction evaluated at the left side of the potential,
    at x = -width/2-dx. 

    Parameters
    ----------
    E : float, energy of the propagating particle.

    dx : float, infinitesimal difference in the position value.

    psi0 : complex float, the value of the wavefunction at x = -width/2
    
    psi_m : complex float, the value of the wavefunction at x = -width/2-dx

    Returns
    -------
    A : complex float, amplitude of the rightwards propagating planewave
    
    B : complex float, amplitude of the leftwards propagating planewave

    """
    #find the coefficients A and B so that we can calculate the reflectance and transmittance 
    q = q_func(E)
    A = (psi_m-psi0*np.exp(-1.j* q * dx )) / (np.exp(+1.j*q * dx) - np.exp(-1.j*q*dx) )
    B = psi0-A
    return A, B


def scattering(E, potential, width, V0, x_num): 
    """
    Returns the transmission and reflection coefficients for the scattering of a particle with energy E through a potential which is 
    determined by the potential function and it has width size and V0 height. The number of position values in the region where the potential
    is non_zero is x_num. 

    Parameters
    ----------
    E : float, energy of the propagating particle.

    potential : function, the potential that is present in the system.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0 : float, height of the potential.

    x_num : integer, the number position values which are in the range (-width/2, width/2).


    Returns
    -------
    R : float, reflection coefficient for this scattering process.
    
    T : float, transmission coefficient for this scattering process.

    """
    #do the whole previous stuff for a given energy value
    
    #define the positions in the systems
    dx = width / x_num #dx is defined in this way
    xs = np.arange(-width/2-dx, width/2, dx) / l0 #lets make the possible position values like this
    dx /= l0 #but we need everything in units of l0 
    width /=l0
    
    #obtain all the k values 
    ks = get_ks(xs = xs, dx = dx, potential = potential, width = width, V0 = V0, E = E)
    
    #define the initial psi values
    psis = [np.exp(-1j*q_func(E)*dx),1]
    
    #carry out the numerov method and obtain the wavefunction
    psis_final = numerov(psis = psis, ks = ks, dx = dx, x_num = x_num)
    
    #calculate the coefficients
    A,B = findAB(E = E, dx = dx, psi0 = psis_final[1], psi_m = psis_final[0])
    
    #find the reflectance and the transmittance
    R = (np.abs(B)**2)/(np.abs(A)**2)
    T = 1/(np.abs(A)**2)
    
    #return their values
    return R,T

def multi_scattering_energies(Es, potential, width, V0, x_num):
    """
    Carries out the numerov method for different energies. 

    Parameters
    ----------
    Es : list of floats, the possible energy values for which the scattering is carried out.
    
    potential : function, the potential that is present in the system.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0 : float, height of the potential.

    x_num : integer, the number position values which are in the range (-width/2, width/2).

    Returns
    -------
    Rs : list of floats, the list of reflection coefficients corresponding to the different energy values in the Es list. 
    
    Ts : list of floats, the list of transmission coefficients corresponding to the different energy values in the Es list. 
    """
    #repeat the the numerov method for different energy values
    
    #list for the R and T values
    Rs = []
    Ts = []
    
    #for every energy value
    for E in Es:
        #get the R and T values
        R, T =  scattering(E = E, potential = potential, width = width, V0 = V0, x_num = x_num)
        
        #add them to the lists
        Rs.append(R)
        Ts.append(T)

    #create numpy arrays out of the lists
    Rs = np.array(Rs)
    Ts = np.array(Ts)

    #return the values
    return Rs, Ts

def multi_scattering_potentials(E, potential, width, V0s, x_num):
    """
    Carries out the numerov method for different potential heights.

    Parameters
    ----------
    E : float, energy of the propagating particle.

    potential : function, the potential that is present in the system.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].

    V0s : list of floats, list that contains the heights of the potential.

    x_num : integer, the number position values which are in the range (-width/2, width/2).

    Returns
    -------
    Rs : list of floats, the list of reflection coefficients corresponding to the different potential height values in the V0 list. 
    
    Ts : list of floats, the list of transmission coefficients corresponding to the different potential height values in the V0 list. 

    """
    #repeat the the numerov method for different energy values
    
    #list for the R and T values
    Rs = []
    Ts = []
    
    #for every energy value
    for V0 in V0s:
        #get the R and T values
        R, T =  scattering(E = E, potential = potential, width = width, V0 = V0, x_num = x_num)
        
        #add them to the lists
        Rs.append(R)
        Ts.append(T)

    #create numpy arrays out of the lists
    Rs = np.array(Rs)
    Ts = np.array(Ts)

    #return the values
    return Rs, Ts


def analytical_curve(E,V0,width):
    """
    Calculate the analytical curves for the transmission coefficients.

    Parameters
    ----------
    E : float, energy of the propagating particle. In units of eV.

    V0 : float, height of the potential.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].


    Returns
    -------
    T : float, analytical value for the transmission coefficient for a square potential. 

    """
    a = width * 1e-9
    
    if E-V0 >= 0 : #classical regime
        k = np.sqrt(2*electron_mass*e*(E-V0))/ hbar
        
        den = 1 + V0**2 / (4 * E * (E-V0) ) * (np.sin(k*a))**2
        
    else:
        k = np.sqrt(2*electron_mass*e*(V0-E))/ hbar
        
        den = 1 + V0**2 / (4 * E * (V0-E) ) * (np.sinh(k*a))**2
        
    return np.real(1/den)

def get_acurve_energies(Es, V0, width):
    """
    Get analytical curves for the different energy values in list Es.

    Parameters
    ----------
    Es : list of floats, the possible energy values for which the scattering is carried out.

    V0 : float, height of the potential.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].


    Returns
    -------
    ac : list of floats, transmission coefficients corresponding to the different energies.

    """
    
    #get the analytical values for the same potential
    ac = []
    
    for E in Es:    
        ac.append(analytical_curve(E = E, V0 = V0, width = width))
        
    ac = np.array(ac)
    
    return ac

def get_acurve_potentials(E, V0s, width):
    """
    Get analytical curves for the different potential heights in the list V0s.

    Parameters
    ----------
    E : float, energy of the propagating particle. In units of eV.

    V0s : list of floats, list that contains the heights of the potential.

    width : float, total width of the potential. The potential is nonzero in the region [-width/2,width/2].


    Returns
    -------
    ac : list of floats, transmission coefficients corresponding to the different potential heights.

    """
    
    #get the analytical values for the same potential
    ac = []
    
    for V0 in V0s:    
        ac.append(analytical_curve(E = E, V0 = V0, width = width))
        
    ac = np.array(ac)
    
    return ac

#Below I defined some functions to plot the curves
def plotter_square_energies(Es, V0, Ts, ac, width, save = False):
    
    fig = plt.figure()

    plt.plot(Es/V0,Ts, label = "numeric")
    plt.plot(Es/V0, ac,  "--", label = "exact")
    
    xt = np.array([0,0.25,0.5,0.75,1])
    plt.yticks(xt)
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    plt.xlabel(r"$E/|V_0|$", fontsize = 16)
    plt.ylabel(r"$T$", fontsize = 16)
    plt.ylim(0,1.1)
    plt.plot([1,1],[0,1], "--", color = "grey")
    
    plt.legend(fontsize = 12)
    
    plt.title(f"Transmission for a square potential well with "+"\n"+
             f"$V_0 = {V0:.0f}$ and $2a = {width:.1f}$",fontsize = 16)
    
    plt.grid()
    
    if save:
        plt.savefig("square_well_plot_es.png",dpi = 800)
    else:
        pass
    
    plt.show()
    
def plotter_square_potentials(E, V0s, Ts, ac, width, save = False):
    
    fig = plt.figure()

    plt.plot(V0s, Ts, label = "numeric")
    plt.plot(V0s, ac,  "--", label = "exact")
    
    xt = np.array([0,0.25,0.5,0.75,1])
    plt.yticks(xt)
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    plt.xlabel(r"$V_0$", fontsize = 16)
    plt.ylabel(r"$T$", fontsize = 16)
    plt.ylim(0,1.1)

    
    plt.legend(fontsize = 12)
    
    plt.title(f"Transmission for a square potential well with "+"\n"+
             f"$E = {E:.1f}$ and $2a = {width:.1f}$",fontsize = 16)
    
    plt.grid()
    
    if save:
        plt.savefig("square_well_plot_pots.png",dpi = 800)
    else:
        pass
    
    plt.show()
    
def plotter_gaussian_energies(Es, V0, Ts, width, save = False):
    
    fig = plt.figure()

    plt.plot(Es/V0,Ts, label = "numeric")
    
    xt = np.array([0,0.25,0.5,0.75,1])
    plt.yticks(xt)
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    plt.xlabel(r"$E/|V_0|$", fontsize = 16)
    plt.ylabel(r"$T$", fontsize = 16)
    plt.ylim(0,1.1)
    plt.plot([1,1],[0,1], "--", color = "grey")
    
    plt.legend(fontsize = 12)
    
    plt.title(f"Transmission for a gaussian potential well with "+"\n"+
             f"$V_0 = {V0:.0f}$ and $2a = {width:.1f}$",fontsize = 16)
    
    plt.grid()
    
    if save:
        plt.savefig("gaussian_well_plot_es.png",dpi = 800)
    else:
        pass
    
    plt.show()
    
def plotter_gaussian_potentials(E, V0s, Ts, width, save = False):
    
    fig = plt.figure()

    plt.plot(V0s, Ts, label = "numeric")
    
    xt = np.array([0,0.25,0.5,0.75,1])
    plt.yticks(xt)
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    plt.xlabel(r"$V_0$", fontsize = 16)
    plt.ylabel(r"$T$", fontsize = 16)
    plt.ylim(0,1.1)
    
    
    plt.legend(fontsize = 12)
    
    plt.title(f"Transmission for a gaussian potential well with "+"\n"+
             f"$E = {E:.1f}$ and $2a = {width:.1f}$",fontsize = 16)
    
    plt.grid()
    
    if save:
        plt.savefig("gaussian_well_plot_pots.png",dpi = 800)
    else:
        pass
    
    plt.show()