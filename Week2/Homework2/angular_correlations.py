import numpy as np
import matplotlib.pyplot as plt

#the strength of the interaction
Int_length = 1.1

def dist(x1,y1,x2,y2):
    """
    Auxiliary function that calculates the usual euclidean distance of two points
    
    inputs:
    -x1,y1: x,y coordinates of the first point
    -x2,y2: x,y coordinates of the second point
    
    output:
    -euclidean distance of the points
    """
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

class Bucket:
    def __init__(self):
        """
        Creates a Bucket object which is used to contain a few particles in the lattice.
        """
        self.positions = [] #array for the positions
        self.part_num = 0
        
    def print_bucket(self):
        """
        Prints the attributes of a Bucket object.
        """
        print("This is a Bucket")
        print("\t"+"number of particles: ", self.part_num)
        print("\t"+"positions: ")
        print(self.positions)
        
    def self_correlation(self): #calculate the angular distribution of position vectors 
        """
        Calculates the correlation of particles within one Bucket.
        
        output:
        -list of angles corresponding to the correlations
        """
        ang_corr = []
        
        for i in range(self.part_num): #for each particle in the Bucket
            [x1, y1] = self.positions[i] #coordinates of the first particle
            for j in range(i+1,self.part_num): #and for each other particle in the Bucket starting from the current one 
                [x2,y2] = self.positions[j] #coordinates of the second particle
                if i != j and dist(x1,y1,x2,y2)<Int_length: #if these are distinct particles and they are closer to each other than the interacion length
                    #(actually by construction of the for cycles it is guaranteed that i never equals j)
                    #then we want to add the angle to the angular distribution list
                    phi = np.arctan2(y2-y1,x2-x1) #angle of the displacement vector wrt to the x axis
                    ang_corr.append(phi) #angle of the displacement vector wrt to the x axis 
                    
                    #if for a correlation between i-j we know phi, then we can calculate the correlation for j-i, by substracting
                    #or adding pi to phi, depending on the sign of phi
                    ang_corr.append(phi-1*np.sign(phi)*np.pi) #this gives the correlation between j and i 
                    #by adding this other term we can safely use the above summation because we will not leave out any angle

                else: #particle self correlation is omitted
                    pass
        #return the angles
        return ang_corr
    
    def pair_correlation(self,neighbour_bucket):
        """
        Function that calculates the correlation between two Bucket objects, but only "in one way".
        That is, if r_i is the position vector of a particle in the first Bucket and r_j is the position 
        of a particle in the second Bucket, then only angle(r_j-r_i) is added to the angular correlations.
        We do this because it is hard to keep track of what has been already calculated and it is easier
        to just calculate everything twice.
        
        inputs:
        -(self) 
        -a Bucket object which is in the neighbourhood of the self Bucket
        
        output:
        -angular correlation list 
        """
        
        ang_corr = []
        
        for i in range(self.part_num): #for each particle in the self bucket
            [x1, y1] = self.positions[i] #position of ith particle in self bucket
            for j in range(neighbour_bucket.part_num): #for each particle in the neighbour bucket
                [x2,y2] = neighbour_bucket.positions[j] #position of the jth particle in neighbour bucket
                
                if dist(x1,y1,x2,y2)<Int_length: #if they are closer to eachother than the interaction length
                    phi = np.arctan2(y2-y1,x2-x1)
                    ang_corr.append(phi) #add the angle to the angular correlation list
                    
        #return the angular correlation between the two buckets
        #note that this angular correlation only contains only the angles calculated from the self bucket towards the neighbour
        #in this sense this is only half of the total correlation between the two buckets
        return ang_corr 
    
class Lattice:
    def __init__(self,filename):
        """
        Creates a Lattice object which is used to contain all the particles.
        
        input:
        -filename of the file containing the positions of the particles
        """
        
        data = np.loadtxt(f"./hw02/{filename}") #load the positions of the particles
        
        #shift the x and y coordinates so that all the positions lie in the positive x-y plane
        #such a shift does not change the distribution of angular correlations
        data[:,0] = data[:,0]-np.min(data[:,0])
        data[:,1] = data[:,1]-np.min(data[:,1])
        
        #We need to find the length of the lattice in the x direction
        Lx = np.max(data[:,0])-np.min(data[:,0]) #size of the lattice in the x direction
        #we need to add an additional small term to Lx such that we can store the particle with the largest x coordinate
        Lx += np.max(data[:,0])/1e8
        
        num = int(Lx/Int_length) #this is the maximal integer number that can be used to divide Lx into num cells such that the linear length of a 
        #cell will be no longer than the Int_length
        
        d = Lx/num # will be the length of a cell
        #and with that being said, actuall num is the number of cells in the x direction
        n = num #number of cells in the x direction
        
        #and we still need the number of cells in the y direction
        m = int((np.max(data[:,1])-np.min(data[:,1]))/d) + 1 #number of cells in the y direction
        
        #create lattice of suitable size which will contain Bucket objects
        bucket_matrix = []
        for i in range(n): #for each cell in the x direction
            bucket_matrix.append([]) 
            for j in range(m): #for each cell in the y direction
                bucket_matrix[-1].append(Bucket()) #put a Bucket object at bucket_matrix[i][j]
        
        #now fill the buckets with the positions of the particles
        for point in data: #for particle in the data
            px = point[0] #x coordinate
            py = point[1] #y coordinate
            i = int(px/d)
            j = int(py/d)
            (bucket_matrix[i][j]).positions.append([px,py]) #add the particle to the correct Bucket
            (bucket_matrix[i][j]).part_num += 1 #increase the number of particles in the bucket
        
        #now the lattice has a matrix of bucket objects which are filled with the corresponding particles
        self.bucket_matrix = bucket_matrix
        
        self.x_length = n #length of the lattice in the x direction (apart from the periodic boundary condition
        self.y_length = m #length of the lattice in the y direction
        self.number_of_particles = len(data) #number of total particles in the system
        self.cell_size = d #length of the cells in the system
        
    def part_num_checker(self):
        """
        Function that finds the maximal number of particles in a Bucket.
        """
        k = 0
        for i in range(len(self.bucket_matrix)):
            for j in range(len(self.bucket_matrix[i])):
                if self.bucket_matrix[i][j].part_num > k:
                    k = self.bucket_matrix[i][j].part_num
                    
        if k<=4:
            print("Particle number okay:"+"\n"+"no bucket contains more than 4 particles")
            print("Maximal number of particles in a bucket:",k)
            
    def lattice_print(self):
        """
        Function that prints out information about the Lattice object.
        """
        print("This is a Lattice")
        print("Lattice size in the")
        print("\t"+"x direction: ",self.x_length)
        print("\t"+"y direction: ",self.y_length)
        print("Number of particles in the Lattice :",self.number_of_particles)
        print("Linear size of the cells: ", self.cell_size)
            
    
    def angular_correlation(self):
        """
        Function that calculates the angular correlation for the lattice.
        
        output:
        -list of angles
        """
        #list that contains all the possible directions where a Bucket can have a neighbour
        #we use this list to iterate through the possible neighbours
        neighbours = [[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]]
        
        #collect the angles here
        total_angular_corr = []
        
        #now calculate the angles
        
        #iterate through all the buckets in the lattice
        for i in range(self.x_length): 
            for j in range(self.y_length): 
                #the bucket we consider is at the position i,j in bucket_matrix
                bucket = self.bucket_matrix[i][j]
                
                #now for each bucket
                
                #first calculate the self correlation
                self_corr = bucket.self_correlation()
                
                #and add this self correlation to the total angular correlation
                total_angular_corr.extend(self_corr)
                
                #now go through all of its neighbours
                for neigh_displ in neighbours: 
                    #calculate the coordinates of the neighbour bucket in the bucket_matrix
                    #warning: we need periodic boundary condition in the x direction
                    #but NOT in the y direction
                    
                    ni = (i+neigh_displ[0])%self.x_length #this yields periodic boundary in the x direction
                    
                    nj = j+neigh_displ[1] #this is just the y coordinate and we have to check whether this is a valid entry in the bucket matrix
                    if nj < 0 or nj >= self.y_length: #if it is not smaller or larger than the allowed values
                        pass
                    
                    else: #then this is a correct neighbor for which we want to calculate the pair-bucket-correlation
                        neighbour_bucket = self.bucket_matrix[ni][nj]
                        
                        #calculate the pair correlation for these two buckets 
                        #again remark: this is only a ONE WAY correlation!
                        pair_corr = bucket.pair_correlation(neighbour_bucket)
                        
                        #add the pair correlation to the total angular correlation
                        total_angular_corr.extend(pair_corr)
        
        #return the total angular correlation
        return total_angular_corr
    
def corr_checker(ang_corr):
    """
    Function that can be used to find fatal mistakes in the correlations. 
    Principle: We know that there is a direct correspondance between the angle for ri-rj and rj-ri. 
    If the angles contained in the list does not have a pair corresponding to the above direct correspondance,
    then we can be sure that something is wrong with the angles.
    
    Of course, by using this function we cannot conclude that the angular correlations are indeed correct,
    but we can easily realize that something is wrong.
    
    Usage of this function is not suggested when working with large data, as the method implemented 
    is rather brute force and it is by no means effective.
    
    input:
    -list of angles corresponding to the calculated angular correlations of a Lattice object
    """
    for angle in ang_corr: #iteart trough all the angles
        pair_angle = angle-1*np.sign(angle)*np.pi #find the pair of the given angle
        b = np.any(np.isclose(ang_corr, pair_angle, atol=1e-8)) #check whether the pair angle is indeed in the list
        if b == False: #if it is not in the list then we already know something is wrong
            print("Correlations are inconsistent!") #print the message and return 
            return None
    print("Correlations are consistent.") #if there were no missing pairs then we are happy and print this nice message
    return None

def histogrammaker(angular_correlation, ang_min, ang_max, bin_num):
    """
    Function that makes a histogram out of the distribution of the angular correlations
    
    inputs:
    -angular_correlation: list of angles corresponding to the calculated angular correlations for a Lattice object
    -ang_min: minimal angle value for the histogram
    -ang_max: maximal angle value for the histogram
    -bin_num: number of bins to be created for the histogram
    
    outputs:
    -vals: list of the values for each bin
    -bins: list containing the position of the bins
    """
    
    vals = np.zeros(bin_num)
    
    width = (ang_max-ang_min)/bin_num #widht of the bin
    
    for ang in angular_correlation:
        i = int((ang-ang_min)/width)
        vals[i] += 1
    bins = (np.arange(bin_num))*width+ang_min
    
    fig = plt.figure()
    plt.bar(bins,vals,width=width,align="edge")
    plt.xlabel("Angle [rad]",fontsize = 12)
    plt.ylabel("Frequency [event/bin]",fontsize = 12)
    plt.show()
    
    return vals,bins