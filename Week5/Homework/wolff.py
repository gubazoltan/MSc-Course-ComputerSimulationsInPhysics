import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

def create_lattice(Lx,Ly):
    """
    Create a lattice object with linear size Lx and Ly in the x and y direction and fills the lattice with random spins pointing either up or down.
    Also creates the corresponding neighbour lists.

    Parameters
    ----------
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.

    Returns
    -------
    lattice : 2d numpy array, contains spins pointing either up or down.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.

    """
    #create lattice and fill it with random spins either pointing up or down
    #reshape the 1d numpy array so it has a form of a lattice
    lattice = np.random.choice([-1,1], size = Lx*Ly).reshape(Ly,Lx)
    
    #create the neigbours using periodic boundary condition
    neighs = []
    
    #iterate through the lattice cells
    for i in range(Lx*Ly):
        #create the positions of the spin in the lattice (2d coordinate)
        mj = i // Lx
        mi = i - mj*Lx
        
        #add the corresponding neighbours
        neighs.append([[(mj-1)%Ly,mi],
                      [(mj+1)%Ly,mi],
                      [mj,(mi+1)%Lx],
                      [mj,(mi-1)%Lx]])
    #return the lattice object and the neighbours
    return lattice, neighs

def find_droplet(spin, lattice, neighs, actspins, checkedspins, Beta, J, Lx, Ly):
    """
    Recursive function that creates the ising droplet of a random spin. Recursive definition is used because finding the droplet can be done efficiently this way. 

    Parameters
    ----------
    spin : length two list of integers, contains the 2d coordinates of the random spin whose ising droplet we are looking for
        
    lattice : 2d array of integers, the lattice that contains the spins pointing either up or down.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.
    
    actspins : 1d list of spins, list of spins (in the 2d representation) which are already added to the ising droplet
    
    checkedspins : 1d list of spins, list of spins (in the 2d representation) which are either added or not added to the list of active spins. These spins have already
            been considered and they must not be tested again
            
    Beta : float, inverse temperature of the system.
    
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.

    Returns
    -------
    actspins : 1d list of spins, list of spins (in the 2d representation) which are already added to the ising droplet
    
    checkedspins : 1d list of spins, list of spins (in the 2d representation) which are either added or not added to the list of active spins. These spins have already
            been considered and they must not be tested again

    """
    #list of spins that have been newly added to the list of active spins. That is, these spins are neighbours of spin and they have been added to the active spins
    newly_added = []
    
    #linear index of the spin 
    i = spin[0]*Lx+spin[1]
    
    #get the value of the spin in the lattice
    spin_val = lattice[spin[0],spin[1]]
    
    #iterate trough the neighbours of the spin
    for nspin in neighs[i]:
        
        #spin value of the neighbour
        nspin_val = lattice[nspin[0],nspin[1]]
        
        #if the value of the two spins is the same and the neighbour is not in checkedspins
        if (nspin_val == spin_val) and ( nspin not in checkedspins ) : 
            
            #we generate a random number and add the spin to the others via this proability
            if np.random.random()< (1-np.exp(-2*Beta*J)): #if the random number is okay
                #we add the spin to the list of active spins and also add it to the newly added list
                newly_added.append(nspin)
                actspins.append(nspin)
            else:
                #else we do nothing
                pass
            #but anyways we add the spin to the list of already checkedspins
            checkedspins.append(nspin)
            
    #now we have added all the neighbours of spin to the list of active spins with a proababilistic weight
    #and now we must repeat this for all the spins in the list of newly added spins
    #this is the reason why we define the function iteratively
    
    #iterate through all the newly added spins
    for new_spin in newly_added:
        #and carry out the same procedure for them
        actspins,checkedspins = find_droplet(spin = new_spin, lattice = lattice, neighs = neighs,
                                             actspins = actspins, checkedspins = checkedspins, Beta = Beta, J = J,
                                             Lx = Lx, Ly = Ly)
        
    #while the above algorithm is being carried out the list of active spins gets bigger and bigger so the ising droplet is being formed by the function
    
    #after we have carried out the whole procedure for every neighbrour of every neighbour and so on we return the list of active spins
    #and the list of already checkedspins
    #obviously we care only about the list of active spins but because of the recursive definition of this function we also need the list of checked spins
    return actspins,checkedspins

def update_lattice(lattice,actspins):
    """
    Function that updates the lattice based on the list of active spins. That is, it flips the spin orientation of the spins 
    which are in the list of active spins.

    Parameters
    ----------
    lattice : 2d numpy array, containing the spins in the system.
    
    actspins : 1d array of spins, contains the list of active spins in the ising droplet.

    Returns
    -------
    lattice : 2d numpy array, containing the spins in the system. The lattice is now updated based on actspins.

    """
    #flip all the active spins
    for spin in actspins:
        #flip the spin
        lattice[spin[0],spin[1]] *= -1
    
    #return the lattice
    return lattice

def single_spin_energy(spin_num, lattice, neighs, J, Lx, Ly):
    """
    Calculate the interaction energy of a single spin in the lattice.

    Parameters
    ----------
    spin_num : integer, linear coordinate of the spin.
    
    lattice : 2d numpy array, containing the spins in the system.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.
    
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.

    Returns
    -------
    single_E : float, interaction energy of a single spin

    """
    
    #single spin energy
    single_E = 0
    
    #coordinates of the single spin
    mj = spin_num // Lx
    mi = spin_num - mj*Lx
    
    #value of the spin
    spin_val = lattice[mj,mi]
    
    #iterate through the neighbours
    for neigh in neighs[spin_num]:
        
        #add the interaction energy between the spins to the single spin energy 
        single_E += -1*J*lattice[neigh[0],neigh[1]]
    single_E *= spin_val
    
    #return the single particle interaction energy
    return single_E

def E0(lattice, neighs, J, Lx, Ly):
    """
    Calculate the initial energy of the whole lattice.

    Parameters
    ----------
    lattice : 2d numpy array, containing the spins in the system.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.
  
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.

    Returns
    -------
    E : float, total energy of the lattice

    """
    #calculate the initial energy of the whole lattice
    
    #calculate the total energy
    E = 0
    
    #iterate through all the spins
    for i in range(Lx*Ly):
        #calculate the single spin energies 
        #take into account the overcounting of the bonds between the spins
        E += single_spin_energy(spin_num = i, lattice = lattice, neighs = neighs, J = J, Lx = Lx, Ly = Ly) / 2
    
    #return the total energy
    return E 

def droplet_energy(lattice, actspins, neighs, J, Lx, Ly):
    """
    Calculates the interaction energy of an ising droplet. That is, sum the interaction energy of all the single spins in the list
    of active spins.

    Parameters
    ----------
    lattice : 2d numpy array, containing the spins in the system.
    
    actspins : 1d array of spins, contains the list of active spins in the ising droplet.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.
  
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.

    Returns
    -------
    E_droplet : float, energy of the ising droplet

    """
    #we dont need to calculate the total energy of the system but only the energy at those points where the spins have been flipped
    #what we can do is that we can go though the active spins that we will flip and calculate the interaction energy of those spins
    #before and after the flip
    #the new energy of the system will be E_new = E_initial - E_before + E_after 
    
    E_droplet = 0
    
    #iterate through all the spins in the list of active spins
    for spin in actspins: 
        #one dimensional coordinate of the active spin
        spin_num = spin[0]*Lx+spin[1]
        
        #calculate the energy of the single spin and add it to the E_droplet energy
        E_droplet += single_spin_energy(spin_num = spin_num, lattice = lattice, neighs = neighs, J = J, Lx = Lx, Ly = Ly)
        
    #return the energy of the ising droplet
    return E_droplet

def wolff_algorithm(lattice, neighs, Beta, J, Lx, Ly, itnum, collect_lattice = True):
    """
    Carry out the wolff algorithm.

    Parameters
    ----------
    lattice : 2d numpy array, containing the spins in the system.
    
    neighs : 2d numpy array, contains neighbouring spins for spin i where i = mj*Lx + mi and mj,mi represents the posiiton of the spin in the lattice.
 
    Beta : float, inverse temperature of the lattice.
    
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.
    
    itnum : integer, the number of iterations of the wolff algorithm we carry out.
    
    collect_lattice : boolean, True (False) if we (don't) want to collect snapshots of the lattice as a function of time.

    Returns
    -------
    Es: 1d list of float, contains the energy of the lattice as a function of time
    
    lat_container: 1d list of 2d arrays (lattices), contains the snapshots of the lattice as a function of time.

    """
    
    #initial energy of the lattice    
    E_initial = E0(lattice = lattice, neighs = neighs, J = J, Lx = Lx, Ly = Ly)
    
    #create a list to contain all the energies
    Es = [E_initial]
    
    #if collect_lattice is TRUE then we want to collect the snapshots of the lattice as a function of time
    if collect_lattice:
        #add a copy of the lattice to the container
        #it is important to add a copy because otherwise the snapshots will not be constant
        lat_container = [lattice.copy()]
    else:
        pass
    
    #now do the simulation for many iterations 
    for i in range(itnum):
        #in every iteration we first need to find a random spin and create an ising droplet around that spin
        #this random spin will be the first active spin in the system
        random_spin = [np.random.randint(Ly), np.random.randint(Lx)]
        
        #add the random spin to the list of randacitve spin and to the list of the checked spins
        actspins = [random_spin]
        checkedspins = [random_spin]
        
        #find the droplet of that given random spin
        actspins, checkedspins = find_droplet(spin = random_spin, lattice = lattice, neighs = neighs, actspins = actspins, checkedspins = checkedspins,
                                            Beta = Beta, J = J, Lx = Lx, Ly = Ly)
        
        #now we calculate the droplet_energy of the active spins before flipping the active spins
        Edroplet_before = droplet_energy(lattice = lattice, actspins = actspins, neighs = neighs, J = J, Lx = Lx, Ly = Ly)
        
        #now update the lattice based on the active spins 
        lattice = update_lattice(lattice = lattice, actspins = actspins)
        
        #now calculate the droplet energy of the system after the spins have been flipped
        Edroplet_after = droplet_energy(lattice = lattice, actspins = actspins, neighs = neighs, J = J, Lx = Lx, Ly = Ly)

        #update the energy of the system
        E_new = Es[-1] - Edroplet_before + Edroplet_after
        Es.append(E_new)
        
        #again if we want to collect lattices then we add a snapshot to lat_container
        if collect_lattice:
            lat_container.append(lattice.copy())        
        else:
            pass
        
    #based on the value of collect_lattice
    if collect_lattice:
        #return the energies of the system and the lattices
        return Es, lat_container
    else:
        #return only the energies
        return Es

def energy_fluctuations(Betas, J, Lx, Ly, itnum):
    """
    Function that finds the energy fluctuations of the system as a function of the inverse temperature

    Parameters
    ----------
    Betas : 1d list of floats, the inverse temperature values for which we calculate the energy fluctuations.
    
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.
    
    itnum : integer, the number of iterations of the wolff algorithm we carry out.

    Returns
    -------
    energy_flucts : 1d numpy array of floats, the variation of the energy for the different beta values

    """
    #create an initial lattice and neighbours 
    initial_lattice, neighs = create_lattice(Lx = Lx, Ly = Ly)
    
    #beta dependent energy fluctuations
    energy_flucts = []
    
    #for each beta value do the wollf algorithm
    for Beta in Betas:
        Es = wolff_algorithm(lattice = initial_lattice.copy(), neighs = neighs, Beta = Beta, J = J,
                                 Lx = Lx, Ly = Ly, itnum = itnum, collect_lattice = False)
        #calculate the variation of the energy for the final 70% of the data points
        energy_flucts.append(np.var(Es[int(itnum*0.3):]))
    
    #make a numpy array
    energy_flucts = np.array(energy_flucts)
    
    #return the fluctuations
    return energy_flucts

def ensemble_av(Betas, J, Lx, Ly, itnum, ens_num):
    """
    Do an ensemble average for the the beta values.

    Parameters
    ----------
    Betas : 1d list of floats, the inverse temperature values for which we calculate the energy fluctuations.
    
    J : float, coupling of neighbouring spins in the lattice
    
    Lx : integer, linear size of the lattice in the x direction.
    
    Ly : integer, linear size of the lattice in the y direction.
    
    itnum : integer, the number of iterations of the wolff algorithm we carry out.
    
    ens_num : integer, the number of iterations characterizing the ensemble average

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #create a lattice
    initial_lattice, neighs = create_lattice(Lx = Lx, Ly = Ly)
    
    #collect the fluctuations here
    energy_flucts = []    
    
    #for each beta value
    for Beta in Betas:
        #fluctuations
        E_var = 0
        
        #do the simulation for ens_num times
        for i in range(ens_num):
            
            #do the time evolution
            Es = wolff_algorithm(lattice = initial_lattice.copy(), neighs = neighs, Beta = Beta, J = J,
                                 Lx = Lx, Ly = Ly, itnum = itnum, collect_lattice = False)
            #add the variation to the collector and divide by number of ens_num
            E_var += np.var(Es[int(itnum*0.4):]) / ens_num
        
        #add the fluctuation
        energy_flucts.append(E_var)
    
    #make a numpy array
    energy_flucts = np.array(energy_flucts)
    
    #return the fluctuations
    return energy_flucts

def anim(latcont):
    """
    Create an animation showing the time evolution of the lattice

    Parameters
    ----------
    latcont : 1d list of lattices, contains the lattice snapshots of a time evolution

    Returns
    -------
    d : animation

    """
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    fig, ax = plt.subplots()
    
    #funcanimation
    def animate(t):
        #dont show all the steps only every 10th
        plt.imshow(latcont[10*t])

    #create the animation with this many frames
    frames = int(len(latcont)/10)
    d = matplotlib.animation.FuncAnimation(fig, animate, frames=frames)
    
    #return the animation
    return d