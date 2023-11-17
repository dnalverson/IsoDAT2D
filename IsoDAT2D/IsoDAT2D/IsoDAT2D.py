# Copyright (c) 2023, Danielle N. Alverson
# All rights reserved.
#
# This software is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for details.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import csv
import dask
import glob as glob
from tifffile import imread, imshow
import warnings
import os
import dask


def attempt(Real_Data, Length, i, init= None, solver = 'cd', beta_loss = 'frobenius', iter = 500):
    # Create an NMF model with specified parameters
    NMF_model = NMF(n_components=i, init = init, solver = solver, beta_loss = beta_loss, max_iter = iter)
    
    # Fit the NMF model to the input data
    NMF_data= NMF_model.fit_transform(Real_Data)
    
    # Extract the factorized matrix components
    fit_compos = NMF_model.components_
    
    # Calculate the reconstruction error of the model
    Q = np.array(NMF_model.reconstruction_err_)*100
    
    # Suppress warnings from the library
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Return the reconstruction error, factorized matrix components, and transformed data
    return Q, fit_compos, NMF_data

def Run_NMF(Real_Data, init= None, solver = 'cd', beta_loss = 'frobenius', itear = 1000):
    """ 
    This function performs NMF (Non-negative Matrix Factorization) on the input data, Real_Data.
    The function takes in the following parameters:
    Real_Data: the input data to be factorized
    init: the initial guess of the factorization (default is None)
    solver: the solver to be used in the factorization (default is 'cd' - Coordinate Descent)
    beta_loss: the beta-divergence loss function to be used (default is 'frobenius')
    itear: the number of iterations to perform in the factorization (default is 1000)

    The function returns a tuple of two items:
        1. A pandas dataframe of the NMF components
        2. The reconstructed NMF data
        
    The function uses Dask for parallel computing, and opens a web browser to display the Dask dashboard. 
    """
    import dask.delayed
    import dask.compute
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from dask.distributed import Client
    client = Client()
    import webbrowser
    webbrowser.open(client.dashboard_link)

    # Initialize variables
    In = init
    Solve = solver
    Beta = beta_loss
    It = itear

    # Delayed computation of the function attempt for each number of components
    jobs = [dask.delayed(attempt)(Real_Data, Real_Data.shape[1], i, In, Solve, Beta, It) for i in range(1, Real_Data.shape[1])]
    calcs = dask.compute(jobs)[0]

    # Convert the results to a numpy array
    calcs = np.array(calcs)

    # Find the minimum beta-divergence between the training data and the reconstructed data
    min_Q = np.min(calcs)
    noc = np.where(calcs == min_Q)
    noc_2 = noc[0]
    number_of_components = noc_2[0] + 1

    # Run the final NMF with the number of components found
    Divergence, compos, NMF_Data_2 = attempt(Real_Data, Real_Data.shape[1], number_of_components, In, Solve, Beta)

    # Print the beta-divergence and number of components used
    print('The beta-divergence between the training data and reconstructed data is',
            Divergence,'%', 'The final number of components used were', number_of_components + 1, '/n')

    # Create a pandas dataframe of the NMF components
    components_dataframe = pd.DataFrame(compos).T

    # Plot the NMF components
    plt.figure(figsize=(10, 10))
    plt.plot(components_dataframe, linewidth=2, alpha=0.7, c='k')
    plt.title('NMF Components')
    plt.ylabel('Intensity')
    plt.xlabel('Q')

    return components_dataframe, NMF_Data_2

def AggCluster(Number_Clusters, data):
    
    """
    A program that will take in the type of scikitlearn clustering algorithm
    desired and the number of clusters as well as the data in a numpy array
    and output the associated clusters with the original data. This will make
    the 'latent' space from the clustering algorithms have more meaning
    
    Input:
    Number_Clusters - int, the number of clusters desired
    data - numpy array, the data to be used for clustering
    
    Output:
    Understanding_data - dictionary, contains the cluster number and the associated
                        int_angle value
    """
    
    # Import the AgglomerativeClustering function from the scikit-learn library
    from sklearn.cluster import AgglomerativeClustering
    
    # Create an AgglomerativeClustering object with the desired number of clusters
    Make_Clusters = AgglomerativeClustering(n_clusters=Number_Clusters, compute_distances=True)
    
    # Fit the model to the data and predict the cluster labels
    y_kmeans = Make_Clusters.fit_predict(data)
    
    # Fit the model to the data
    information = Make_Clusters.fit(data)
    
    # Get the distances parameter
    parameter = information.distances_

    # Initialize the variables for looping
    x = 0
    Understanding_data = {"Cluster_Number":[], "Int_Angle":[]}
    
    # Loop through each data point and add the cluster number and int_angle value to the dictionary
    while x < len(data):
        Understanding_data["Cluster_Number"].append(y_kmeans[x])
        Understanding_data["Int_Angle"].append(data[x])
        x = x+1
        
    # Loop through each cluster and create a separate plot for each cluster
    q = 0
    while q < Number_Clusters:
        z = 0
        plt.figure()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Agglomerative Clustering' + ' ' + str(q))

        # Loop through each data point and plot only the points belonging to the current cluster
        while z < len(data):
            if Understanding_data["Cluster_Number"][z] == q:
                plt.plot(Understanding_data["Int_Angle"][z], label='Component' + str(z))
                
            z = z + 1
        q = q + 1
        
    # Return the dictionary of cluster numbers and int_angle values
    
    return Understanding_data

def Combine_Cluster(Understanding_data):
    """
        A program that will take in a user specified cluster or clusters of choice, from the Agglomerative 
        cluster function and average the signals for one array of data. This data will then be
        smoothed in a different function called smooth_data.
        
        Input: 
        Understanding_data - dictionary, contains the cluster number and the associated
        
        Output:
        Identified_signal - averaged numpy array, the averaged signal from the cluster of interest
    """
    cluster_of_interest = []
    keep_going = True
    cluster_numbers = np.array(Understanding_data['Cluster_Number'])
    
    while keep_going == True:
        print('Please enter the cluster number of interest')
        for i in range(0, len(cluster_numbers)):
            print('Cluster ', i+1)
        selected_cluster = int(input("Enter the cluster number of interest: ")) - 1
        cluster = np.where(cluster_numbers == selected_cluster)
        cluster_data = cluster[0]
        cluster_of_interest.append(cluster_data)
        keep_going = bool(input("Would you like to add another cluster? (True/False)"))
    
    arrayed_cluster = np.array(cluster_of_interest)
    Identified_signal = np.mean(arrayed_cluster, axis = 0)
    
    plt.figure(figsize = (10,10))
    plt.plot(Identified_signal)
    plt.x_label('Q')
    plt.ylabel('Intensity')
    plt.title('Averaged Signal')
    
    return Identified_signal