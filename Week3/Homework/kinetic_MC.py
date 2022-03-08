import numpy as np

def entry_transform(number, N):
    """
    Function that generates the position of the particle in the lattice matrix based on the input 'number'. 

    Parameters
    ----------
    number : Integer value. Number is the 1D position of a particle that needs 
            to be converted into a 2D posoition.
    N : Integer value. Linear length of the lattice containing the particles. The lattices has N*N unit cells.

    Returns
    -------
    i : Integer value. Row index of the entry in the lattice matrix characterizing 
    the position of the particle.
        
    j : Integer value. Column index of the entry in the lattice matrix characterizing 
        the position of the partice.

    """
    
    i = int(number/N)
    j = number-i*N
    return i,j

def create_neighs():
    """
    Function that creates all the possible displacement vectors that characterize the nearest neighbours 
        of a unit cell i,j. The relative position of the neighbours depend on whether i is an even or an
        odd integer, because the system is a tringular lattice.

    Returns
    -------
    neighs : List of lists. neighs[0] (neighs[1]) contains the relative poisiton vectors of the neighbours of 
        unit cell i,j where i is an even (odd) integer.

    """
    even_neighbours = [[0,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0]]
    odd_neighbours = [[0,1],[1,0],[1,1],[0,-1],[-1,0],[-1,1]]
    neighs = [even_neighbours,odd_neighbours]  
    return neighs

class Lattice:
    """
    Class describing the triangular lattice of ours.
    """
    def __init__(self, N, particle_number, Beta):
        """
        Function that creates a Lattice object with linear size N, particle_number of particles
            and temperature Beta.

        Parameters
        ----------
        N : Integer value. N is the linear size of the triangular lattice.
        
        particle_number : Integer value. Number of particles distributed randomly on 
            the N*N triangular lattice.
            
        Beta : Float value. Inverse temperature of the system.

        Returns
        -------
        None.

        """
        #create the empty lattice
        lattice_matrix = np.zeros(N*N).reshape((N,N))

        #fill the lattice with particle_number particles
        part_positions = np.random.choice(a = np.arange(N*N), size = particle_number, replace = False)

        #now transform the particle positions to fill in the lattice matrix
        for k in range(particle_number):
            i,j = entry_transform(part_positions[k],N)
            lattice_matrix[i,j] = 1 #place particle in the correct position
        
        #add attributes to the Lattice object
        self.lattice_matrix = lattice_matrix 
        self.size = N 
        self.particle_number = particle_number
        self.temp = Beta
        

    def int_energy(self, i, j):
        """
        Function that calculates the interaction energy of unit cell characterized by integers i,j
            in the lattice matrix. Needed to evaluate the transition rates

        Parameters
        ----------
        i : Integer value. Row index of the entry in the lattice matrix characterizing 
            the position of the particle.
        
        j : Integer value. Column index of the entry in the lattice matrix characterizing 
            the position of the partice.

        Returns
        -------
        energy : Integer value. The number of electrons in the neighbourhood of unit cell i,j.

        """
        #all possible neighbours of the unit cell
        neighs = create_neighs()

        #linear size of the lattice
        N = self.size
        
        #calculate the energy of the electron configuration near unit cell i,j
        energy = 0
        #iterate through the neighbours
        for neigh in neighs[i%2]:
            #absolute position of the neighbours of unit cell i,j
            neigh_i, neigh_j = (i+neigh[0])%N, (j+neigh[1])%N
            
            #add energy for every electron in the neighbourhood of unit cell i,j 
            energy += self.lattice_matrix[neigh_i,neigh_j]
            
        #return the energy
        return energy
        
    def new_configurations(self):
        """
        Function that finds all possible electron configurations of the system and the corresponding
            transition rates. The transition rates are not yet normalized. 

        Returns
        -------
        new_configs : List of lists. new_configs[i] contains the i-th the data we need to describe the i-th
            possible new configuration of the system. new_configs[i][0] and new_configs[i][1] contains the
            row and column index of the electron that hops to a new empty unit cell characterizde by 
            row and column indices new_configs[i][2] and new_configs[i][3]. new_configs[i][4] and
            new_configs[i][5] contains the energy corresponding to the old and new configuration of the system
            respectively.
            
        rates : Float value. Transition rates characterizing the possible hoppings in the system. rates[i]
        contains the transition rate corresponding to the transition described by new_configs[i].

        """
        #all possible neighbours of the unit cell
        neighs = create_neighs()
        
        #get information about the Lattice object
        Beta = self.temp
        N = self.size
        lattice = self.lattice_matrix
        
        #list all the possible configurations and the transition rates here
        new_configs = []
        rates = []
        
        #iterate through the lattice
        for i in range(N): #rows
            for j in range(N): #columns
                
                if lattice[i,j] == 1: #there is a particle in the lattice
                    E_old = self.int_energy(i,j) #calculate the energy of the old configuration
                    
                    for neigh in neighs[i%2]: #iterate through the neighbours
                    
                        #absolute position of the neighbouring site in the lattice
                        neigh_i, neigh_j = (i+neigh[0])%N, (j+neigh[1])%N 
                        
                        #then if the neighbour is empty we can hop into it
                        if lattice[neigh_i,neigh_j] == 0:
                            
                            #calculate the energy for this empty neighbouring site
                            #we have to subtract 1 from this energy because we also count 
                            #that particle that is currently in unit cell i,j
                            E_new = self.int_energy(neigh_i,neigh_j)-1
                            
                            #rate corresponding to the transition
                            rate = np.exp(-(E_old-E_new)*Beta)
                            
                            #add this new configuration to the possible new configurations
                            new_configs.append([i,j,neigh_i,neigh_j,E_old,E_new])
                            rates.append(rate)
                            
        return new_configs, rates
    
    def update_lattice(self, i, j, new_i, new_j):
        """
        Function that performs the hopping of exactly one electron in the lattice from unit cell i,j to
            unit cell new_i, new_j. The transition corresponding to this process is chosen randomly out
            of all other possible transitions.

        Parameters
        ----------
        i : Integer value. Row index of the entry in the lattice matrix characterizing 
            the old position of the particle.
        
        j : Integer value. Column index of the entry in the lattice matrix characterizing 
            the old position of the partice.
            
        new_i : Integer value. Row index of the entry in the lattice matrix characterizing 
            the new position of the particle.
        
        new_j : Integer value. Column index of the entry in the lattice matrix characterizing 
            the new position of the partice.

        Returns
        -------
        None.

        """
        
        #get the lattice matrix of the Lattice object
        lattice = self.lattice_matrix

        #destroy particle at site i,j
        lattice[i, j] = 0

        #and recreate the particle at site new_i,new_j
        lattice[new_i, new_j] = 1

        #update the lattice matrix
        self.lattice_matrix = lattice


    def find_transition(self, new_configs, rates):
        """
        Function that randomly chooses exactly one transition process out of all possible processes.

        Parameters
        ----------
        new_configs : List of lists. new_configs[i] contains the i-th the data we need to describe the i-th
            possible new configuration of the system. new_configs[i][0] and new_configs[i][1] contains the
            row and column index of the electron that hops to a new empty unit cell characterizde by 
            row and column indices new_configs[i][2] and new_configs[i][3]. new_configs[i][4] and
            new_configs[i][5] contains the energy corresponding to the old and new configuration of the system
            respectively.
            
        rates : Float value. Transition rates characterizing the possible hoppings in the system. rates[i]
        contains the transition rate corresponding to the transition described by new_configs[i].
        
        Returns
        -------
        None.

        """
        
        #total rate or "partition function"
        tot_rate = sum(rates)

        #get a random value between 0 and tot_rate
        u = np.random.random()
        val = u*tot_rate

        #start subtracting the rates from val, when val becomes negative, the l of the current iteration
        #is the index of the process that will happen
        l = -1
        while val > 0:
            l += 1
            val -= rates[l]
        #l is now the process we want to carry out
        
        #we need to carry out the l-th transition
        #new configuration will be this
        chosen_config = new_configs[l] 
        
        #attributes of the chosen transition
        i, j, new_i, new_j = chosen_config[0], chosen_config[1], chosen_config[2], chosen_config[3]
        
        #update the lattice according to the new configuration
        self.update_lattice(i = i, j = j, new_i = new_i, new_j = new_j)

    def correlations(self):
        """
        Function that calculates the density-density correlations in the lattice.

        Returns
        -------
        corr/2: Integer value. Total density-density correlations in the triangular lattice.
            It is indeed an integer, because we evaluate every correlation twice.

        """
        
        #all possible neighbours of the unit cell
        neighs = create_neighs()

        #density-density correlations will be summed here
        corr = 0

        #get information about the lattice
        N = self.size
        lattice = self.lattice_matrix
        
        #iterate through the lattice
        for i in range(N):
            for j in range(N):
                for neigh in neighs[i%2]: #iterate through the neighbours
                    neigh_i, neigh_j = (i+neigh[0])%N, (j+neigh[1])%N #position of the neighbour 
                    corr += lattice[i,j]*lattice[neigh_i,neigh_j] #add the correlations 

        return corr/2 #avoid double counting
    
def simulate_TD_correlations(Lattice, it_num):
    """
    Function that can be used to obtain the time evolution of the density-density correlation in the
        system.

    Parameters
    ----------
    Lattice : Lattice object. The Lattice object whose time-dependent correlations we want to calculate.
    
    it_num : Integer value. The number of steps of the kinetic Monte Carlo algorithm. 

    Returns
    -------
    corrs: Numpy array. Array that contains the density-density correlation of the system for each 
        iteration.

    """
    #find the time dependence of the correlations in the lattice
    corrs = []
    
    #initial correlation
    corrs.append(Lattice.correlations())
    
    #do it_num iterations
    for n in range(it_num):
        
        #create a new configuration with transition rates
        newconfigs, rates = Lattice.new_configurations()
        
        #update the lattice according to the transition rates
        Lattice.find_transition(new_configs = newconfigs, rates = rates)
        
        #add the new correlations to the corrs list
        corrs.append(Lattice.correlations())
        
    #return the correlations
    return np.array(corrs)

def simulate(Lattice, it_num): 
    """
    Function that is used to obtain the final density-density correlation of the system. Does not calculate
        calculate the intermediate correlations in the system.

    Parameters
    ----------
    Lattice : Lattice object. The Lattice object whose time-dependent correlations we want to calculate.
    
    it_num : Integer value. The number of steps of the kinetic Monte Carlo algorithm.

    Returns
    -------
    Tcorrelation: Integer value. Final correlation of the system.

    """
    
    #do it_num iterations
    for n in range(it_num):
        #create a new configuration with transition rates
        newconfigs, rates = Lattice.new_configurations()
        
        #update the lattice according to the transition rates
        Lattice.find_transition(new_configs = newconfigs, rates = rates)
      
    #return the final correlation
    return Lattice.correlations()

def transition_point_finder(Betas, N, particle_number, it_num):
    """
    Function that can be used to find the phase transition point of the triangular lattice.

    Parameters
    ----------
    Betas : List of floats. Inverse temperatures for which we carry out the time evolution.
    
    N : Integer value. Linear size of the triangular lattice
    
    particle_number : Integer value. Number of particles that we distribute on the lattice.
    
    it_num : Integer value. The number of steps of the kinetic Monte Carlo algorithm.

    Returns
    -------
    final_corr: Numpy array of floats. Array containing the final density-density correlation of the lattice
        on a given temperature characterized by the inverse temperatures Betas.

    """
    
    #contain the final correlations here
    final_corr = []
    
    #we want to use the same initial layout of particles for all the cases
    initial_lattice = Lattice(N = N, particle_number = particle_number, Beta = Betas[0])
    
    for Beta in Betas:
        #create a new lattice with Beta temperature
        current_lattice = Lattice(N = N, particle_number = particle_number, Beta = Beta)
        
        #set the lattice matrix of the new lattice to the lattice matrix of the initial lattice
        current_lattice.lattice_matrix = initial_lattice.lattice_matrix
        
        #get the final correlation of the current lattice for the given temperature
        cors = simulate(Lattice = current_lattice, it_num = it_num)
        
        #append the final correlation of the lattice to the list
        final_corr.append(cors)
        
    #return the density-density correlations as a function of the inverse temperature divided by the 
    #number of particles in the system
    return np.array(final_corr)/particle_number