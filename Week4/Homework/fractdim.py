import numpy as np
import netpbmfile as npf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def divgen(num):
    """
    Calculates the divisors of a number.

    Parameters
    ----------
    num : integer, the number whose divisors we want to calculate.

    Returns
    -------
    divsors : numpy array of integers, the integers which are the divisors of the number num.

    """
    
    #collect the divisors in this array
    divsors = []
    
    #starting value of the iteration variable
    i = 1

    #if i is smaller than the number and it divides the number we add it to the divisors
    while i < num:
        if num % i == 0:
            divsors.append(i)
        
        i += 1
        
    #transform divsors into a numpy array
    divsors = np.array(divsors)
    
    #return the divisors
    return divsors

def single_boxing(fractal, box_length, fsize_x, fsize_y, ones):
    """
    Carries out the boxing for a fractal with box length characterized by box_length.

    Parameters
    ----------
    fractal : 2d numpy array of zeros and ones, zeros denote the object while ones denote the background.
    
    box_length : integer, length of the box we use for the boxing.
    
    fsize_x : length of the fractal in the x direction.
    
    fsize_y : length of the fractal in the y direction.
       
    ones : 2d numpy array of ones, useful when someone calculates the number of boxes needed to cover the fractal.

    Returns
    -------
    num_boxes : integer, the number of boxes needed to cover the fractal using the given box length.
    """
    
    #number of boxes needed in the x and y direction to cover the whole 2d plane that contains the fractal
    fl_x = (fsize_x // box_length) + 1
    fl_y = (fsize_y // box_length) + 1
    
    #number of boxes needed to cover
    num_boxes = 0
    
    for j in range(fl_y): #iterate through the num of boxes needed in the y direction
        for i in range(fl_x): #iterate through the number of boxes needed in the x direction
            fract_sum = np.sum(fractal[i*box_length:(i+1)*box_length,j*box_length:(j+1)*box_length])
            ones_sum = np.sum(ones[i*box_length:(i+1)*box_length,j*box_length:(j+1)*box_length])
            if fract_sum != ones_sum: 
                num_boxes += 1
    return num_boxes

def fract_dim(filename,f,l):
    """
    Calculates the dimension of the fractal using the box covering algorithm.

    Parameters
    ----------
    filename : string, name of the file that contains the fractal as a .pbm file.
    
    f: integer, the first value of blengths that we consider when we want to find the fractal dimension
    
    l: integer, the last value of blengths that we consider when we want to find the fractal dimension

    Returns
    -------
    dim : integer, dimension of the fractal.
    
    blengths : array of integers, lengths of the boxes that were used to cover the fractal.
    
    bnums : number of boxes needed to cover the fractal using the corresponding box lengths in blengths.

    """
    fractal = np.array(npf.imread("./hf04/"+filename+".pbm"))
    
    #height and width of the fractal 
    #height of the fractal is y while width is x
    fsize_y = len(fractal)
    fsize_x = len(fractal[0])
    
    #creates a background with the same size as the fractal
    #this will ne useful when calculating the number of boxes needed to cover the fractal
    ones = np.ones(fsize_x*fsize_y).reshape((fsize_y,fsize_x))
    
    #list that contains the box lengths for which we do the covering
    blengths = (divgen(fsize_x))
    
    #collect the number of boxes needed to cover the fractal here
    bnums = []
    
    #for each box length
    for blength in blengths:
        
        #find the number of boxes needed to cover the fractal
        bnum = single_boxing(fractal = fractal, box_length = blength, fsize_x = fsize_x, fsize_y = fsize_y, ones = ones)
        bnums.append(bnum)
    
    #create numpy array from box numbers
    bnums = np.array(bnums)
    
    #do linear regression for the Bnums(blengths) data
    reg = LinearRegression()
    
    #we do the fitting only for elements in blengths[f:l]
    reg.fit(np.log(blengths).reshape(-1,1)[f:l], np.log(bnums)[f:l])
    
    #the fractal dimension is predicted by the linear regression model
    dim = -1*reg.coef_
    
    #show how good the linear regression is based on the fitted data
    pred = reg.predict(np.log(blengths).reshape(-1,1)[f:l])
    
    #draw a figure for the fitting
    fig=plt.figure()
    plt.plot(np.log(blengths),np.log(bnums), 'o')
    plt.plot(np.log(blengths)[f:l], pred)
    plt.ylabel(r'$\mathrm{ln(N_{box})}$', fontsize = 16)
    plt.xlabel(r'$\mathrm{ln(Length_{box})}$', fontsize = 16)
    #save the figure
    plt.savefig(filename + "_plot.png")
    
    #return the dimension of the fractal, the box lengths and the number of boxes needed to cover the fractal
    return dim,blengths,bnums
