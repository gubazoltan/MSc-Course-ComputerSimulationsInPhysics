import networkx.generators.random_graphs as graphGen
import networkx as nx
import numpy as np

def Dijkstra_algorithm(graph, source, N):
    """
    Function that carries out the Dijkstra algorithm on a graph where the source and the number of nodes
    in the graph are explicitly given.

    Parameters
    ----------
    graph : networkx graph, the graph on which the Dijkstra algorithm will be carried out.
    
    source : integer, the node which functions as the source in the algorithm.
    
    N : integer, the number of nodes in the graph.

    Returns
    -------
    graph : networkx graph, the graph which is the result of the Dijkstra algorithm. That is, each node 
    obtained a number which represents its distance from the source.

    """
    #unvisited nodes at the beginning
    unvisited = list(range(graph.number_of_nodes()))

    #set the initial distance to N+1
    #which is obviously larger than the possible largest distance in a graph of N nodes
    #for a graph of N nodes the maximal distance achievable would be N-1, which occurs if the connection between the nodes is linear
    initial_distance = N + 1
    
    #set the initial distance to infinite for every node which is not the source
    nx.set_node_attributes(graph, initial_distance, "distance")
    #the source should have distance 0
    graph.nodes()[source]["distance"] = 0
    
    #now carry out the algorithm itself
    
    #while there are still unvisited cities
    while len(unvisited) > 0:
        #find the node which is the closest to the source
        #initial closest is set to unvisited[0]
        closest = unvisited[0]
        
        #for every node in the unvisited list
        for node in unvisited:
            #if node is closer to the source than 
            if (graph.nodes()[node]["distance"]  < graph.nodes()[closest]["distance"] ):
                #set new closest node
                closest = node
        
        #now remove the closest from the list of unvisited nodes
        unvisited.remove(closest)
        
        #now iterate through the neighbours of the closest node 
        #and change their distance from the source
        for neighbour in graph.adj[closest]:
            #update the distance of the neighbour if its current distance from the source is larger than the new distance would be
            graph.nodes()[neighbour]["distance"] = min(graph.nodes()[closest]["distance"] + 1, graph.nodes()[neighbour]["distance"])
            
    #return the graph where now the distances are correctly calculated
    return graph

def multi_dict_average(dicts):
    """
    Function that takes a list of dictionaries as its input and then calculates the average
    of the values in the dictionary.

    Parameters
    ----------
    dicts : list of dictionaries, the dictionaries that contain the average distances for each degree.

    Returns
    -------
    average : dictionary, a dictionary in which every key is an integer representing the degree of a node 
    and the corresponding value represents the average distance measured from a node of the given degree
    in a Barabás-Albert graph.

    """
    #collect the averages here
    average = {}
    
    #for each dictionary in the list
    for dct in dicts:
        
        #for each key and value in the dictionary
        for k, v in dct.items():
            
            #if the key is not in the dictionary then we should add it to it
            if k not in average:
                
                #and set its value to zero
                average[k] = 0
                
            #and update it anyways
            average[k] = average[k] + v
            
    #take the average of the distances
    average = {k: v / len(dicts) for k, v in average.items() }
        
    #return a dictionary in which now for each node we have the averaged distances
    return average

def calculate(m, N, itnum):
    """
    Function that carries out the task. That is, for each iteration it creates a Barabás-Albert graph of
    size N and order m, for which it carries out the Dijsktra algorithm for all possible sources. Then for
    each source it calculates the average distance of the other nodes measured from the source of a given degree.
    The degree of the source and the average distance is then attached to a dictionary. The final dictionary will contain
    all the degrees that occur in the graph as keys and the corresponding values are the average distances.
    Then this dictionary is added to a list for each iteration. Finally, an average is taken for each degree.

    Parameters
    ----------
    m : integer, order of the graph.
    
    N : integer, number of nodes in the graph.
    
    itnum : integer, the number of iterations to overcome statistical fluctuations.

    Returns
    -------
    averages : dictionary, a dictionary in which every key is an integer representing the degree of a node 
    and the corresponding value represents the average distance measured from a node of the given degree
    in a Barabás-Albert graph for itnum number of different graphs.

    """
    
    #collect the dictionaries containing the averages here
    it_dicts = []
    
    for i in range(itnum):   
        #create a graph on which we work for this iteration
        graph = graphGen.barabasi_albert_graph(N, m)
        
        #create a dictionary in which we collect the average distances for each degree
        degree_dict = {}
        
        #get all the degrees that show up in the graph
        _, degrees = zip(*graph.degree)
        
        #for each degree add a list to degree_dict
        for degree in degrees: 
            degree_dict[degree] = []
        
        #now go over all the possible sources
        for source in graph.nodes():  
            
            #carry out the Dijkstra algorithm to find the distances in the graph measured from the source node
            solved_graph = Dijkstra_algorithm(graph = graph.copy(), source = source, N = N)      
            
            #now calculate the average distance measured from the source node in the graph
            avdist = np.average(list((nx.get_node_attributes(G = solved_graph, name = "distance" )).values()))
            
            #degree of the source
            source_degree = graph.degree[source]
            
            #add the average distance to the list in degree_dict which corresponds to the degree of the source
            degree_dict[source_degree].append(avdist)
            
        #now we need to take the average of the lists which correspond to the degree
        #for each degree in degrees
        for degree in degrees:
            
            #calculate the average of the list containing the distances which corresponds to the given degree
            #basically we exchange each list with its average
            degree_dict[degree] = np.average(degree_dict[degree])
            
        #now add this dictionary of average of averages to the it_dicts list
        it_dicts.append(degree_dict)
        
    #now we need to average out these dictionaries
    averages = multi_dict_average(it_dicts)
    
    #and return the final averaged dictionary
    return averages