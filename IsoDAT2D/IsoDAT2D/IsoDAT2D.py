# Copyright (c) 2024, Danielle N. Alverson
# All rights reserved.
#
# This software is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for details.

import SimDAT2D as sim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.decomposition import NMF
import dask
import glob as glob
from tifffile import imread, imshow
import warnings
import dask
import nimfa
import pyFAI
from sklearn.cluster import AgglomerativeClustering
import masking

def attempt(Real_Data, Length, i, init= None, solver = 'cd', beta_loss = 'frobenius', iter = 500):
    NMF_model = NMF(n_components=i, init = init, solver = solver, beta_loss = beta_loss, max_iter = iter)
    NMF_data= NMF_model.fit_transform(Real_Data)
    fit_compos = NMF_model.components_
    Q = np.array(NMF_model.reconstruction_err_)*100
    warnings.filterwarnings("ignore", category=FutureWarning)
    return Q

def attempt2(Real_Data, Length, i, init= None, solver = 'cd', beta_loss = 'frobenius', iter = 500):
    NMF_model = NMF(n_components=i, init = init, solver = solver, beta_loss = beta_loss, max_iter = iter)
    NMF_data= NMF_model.fit_transform(Real_Data)
    fit_compos = NMF_model.components_
    Q = np.array(NMF_model.reconstruction_err_)*100
    warnings.filterwarnings("ignore", category=FutureWarning)
    return Q, fit_compos, NMF_data


def Run_NMF(Real_Data, init= None, solver = 'cd', beta_loss = 'frobenius', itear = 1000, show = False):
    
    """" Comparison of multiple components without manually comparing
    multiple components all at once. There are a few ways that this can be done. One way
    is to take the residuals of the datasets with themselves to see which is the closests to the 
    "correct" component. This may only be done on sample data potentially. Could include a 
    simulated dataset of what the standing component should look like give or take some 
    variations in the data. When the residuals are taken if it is less than some given 
    percentage the would be the dataset/NMF component to use further. There may be multiple
    thus having to go in manually to find differences. 
    
    To do this, will need to import the simulated XRD pattern from VESTA and then take the residual differences
    of each component and compared to the simulated pattern. Will have the program spit out the compnents that
    meet the cutoff. """

    In = init
    Solve = solver
    Beta = beta_loss
    It = itear
    

    jobs = [dask.delayed(attempt)(Real_Data, Real_Data.shape[1], i, In, Solve, Beta, It) for i in range(1, Real_Data.shape[1])]
    #jobs = [dask.delayed(attempt)(Real_Data, Real_Data.shape[1], i, In, Solve, Beta, It) for i in range(1,20)]
    calcs = dask.compute(jobs)[0]

    calcs = np.array(calcs)
    min_Q = np.min(calcs)
    noc = np.where(calcs == min_Q)
    noc_2 = noc[0]
    number_of_components = noc_2[0] +1

    Divergence, compos, NMF_Data_2 = attempt2(Real_Data, Real_Data.shape[1], number_of_components, In, Solve, Beta)
            
    
    print('The beta-divergence is: ', Divergence, '%\n','The final number of components used were',number_of_components+1, '\n') 
    
    
    m = pd.DataFrame(compos)
    m = m.T
    
    if show == True:
        plt.figure(figsize = (5,5))
        colors = plt.cm.magma(np.linspace(0,1, number_of_components))
        i = 0
        while i < number_of_components:
            plt.plot(m[i], c = colors[i], alpha = 0.7)
            i = i+1
    
    return m,NMF_Data_2, min_Q

#masking algoirthm to create masks for the data

def make_masks(array, slices, offset = 5, width=.5, gits = False):
    masks = []
    mask_2048 = np.zeros((2048, 2048), dtype=bool)
    mask_2048[1024:] = True
    for i in slices:
        masks.append(masking.generate_mask_slices(array, width, i, offset = offset))
        print('Mask with {} slices created'.format(i))
        if gits == True:
            gits_masks = []
            for i in range(len(masks)):
                masks_p = masks[i] + mask_2048
                gits_masks.append(masks_p)
                plt.imshow(masks_p)
    return gits_masks

def AggCluster(Number_Clusters, data):
    
    """A program that will take in the type of scikitlearn clustering algorithm
        desired and the number of clusters as well as the data in a numpy array
        and output the associated clusters with the original data. This will make
        the 'latent' space from the clustering algorithms have more meaning"""

    from sklearn.cluster import AgglomerativeClustering
    Make_Clusters= AgglomerativeClustering(n_clusters = Number_Clusters, compute_distances=True)
    y_kmeans = Make_Clusters.fit_predict(data)
    information = Make_Clusters.fit(data)
    parameter = information.distances_

    x = 0
    Understanding_data = {"Cluster_Number":[], "Int_Angle":[]};
    while x < len(data):
        Understanding_data["Cluster_Number"].append(y_kmeans[x])
        Understanding_data["Int_Angle"].append(data[x])
        x = x+1
        
    # Create an empty list to store the data
    data_list = []
    
        
    q = 0
    while q < Number_Clusters:
        z = 0
        plt.figure(figsize=(5,5))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Agglomerative Clustering'+' ' +str(q))

        while z < len(data):
            if Understanding_data["Cluster_Number"][z] == q:
                plt.plot(Understanding_data["Int_Angle"][z], label = 'Component'+str(z))
            z = z+1
    
        # Check if the plot looks good if the plot looks good, append the data to the list
        plt.show(block = False)
        plt.pause(0.1)
        
        if input("Do the identified components look like an isotropic scattering signal? (y/n)") == 'y':
            i = 0
            while i < len(data):
                if Understanding_data["Cluster_Number"][i] == q:
                    data_list.append(Understanding_data["Int_Angle"][i])
                i = i+1
                
        q = q+1
    
    return Understanding_data, data_list


def smooth_components(Identified_components, filter_strength = 2, show = False):
    '''A function that will smooth the components identified from the agglomerative clustering algorithm'''
    
    # Importing required library
    from scipy.signal import savgol_filter
    
    # Defining a dictionary that maps filter strength to the number of points for the smoothing window
    strength_to_points = {1: 3, 2: 5, 3: 7, 4: 11, 5: 15}
    
    # Retrieving the number of points for the smoothing window based on the filter strength provided
    points = strength_to_points.get(filter_strength)

    # Applying Savitzky-Golay filter to smooth the identified components by taking their average along the columns
    smoothed_compos = savgol_filter(np.mean(Identified_components, axis = 0), points, 1)
    
    # If show argument is True, plotting the original components, their average, and the smoothed components
    if show == True:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax1.plot(Identified_components.T, c = 'k', linewidth = 2, alpha = 0.7)
        ax1.set_title('Identified Components')
        ax1.set_ylabel('Intensity')
        ax2.plot(np.mean(Identified_components, axis = 0), c = 'r', linewidth = 2, alpha = 0.7)
        ax2.set_ylabel('Intensity')
        ax2.set_title('Mean of Identified Components')
        ax3.plot(smoothed_compos, c = 'g', linewidth = 2, alpha = 0.7, label = 'Smoothed Component')
        ax3.scatter(np.arange(len(np.mean(Identified_components, axis = 0))), np.mean(Identified_components, axis = 0), c = 'r', s = 10, 
                    label = 'Mean Component')
        ax3.set_xlabel('Data Points')
        ax3.set_title('Smoothed Component')
        ax3.set_ylabel('Intensity')
        ax3.legend()
    
    # Returning the smoothed components
    return smoothed_compos

import pyFAI.azimuthalIntegrator as AI
import pyopencl.array as cla
import time
import cv2

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@time_function
def image_rotation(image, angle, show = False):
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    if show == True:
        #display the rotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_image, cmap='viridis')
        plt.title("Rotated Image")
        plt.show()
    return rotated_image

@time_function
def rotate_integrate_image_gpu(combined_image,angle_of_rotation, distance, wavelength, resolution = 3000, mask = None, show = True, radial_range = None):
    """
    This function integrates the combined image using the azimuthal integrator and displays the 1D image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
    """
    data = {}
    #initialize the azimuthal integrator
    
     # Initialize the detector
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()
    
    target = (0,0)
    ai = AI.AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)
    #initialize engine
    res0 = ai.integrate1d(combined_image, resolution, radial_range = radial_range, unit = 'q_A^-1', mask = mask, method=("bbox", "csr", "opencl", target))
    
    # Get the engine from res0
    engine = ai.engines[res0.method].engine
    omega = ai.solidAngleArray()
    omega_crc = engine.on_device["solidangle"]
    for i in range(0, 360, angle_of_rotation):
        # start_time = time.time()
        #rotate the mask for the combined image
        rotated_image = image_rotation(combined_image, i)
    
        rotated_image_d = cla.to_device(engine.queue,rotated_image)
    
        res1 = engine.integrate_ng(rotated_image_d, solidangle=omega, solidangle_checksum=omega_crc)
        # print(time.time() - start_time)
        data[i] = res1.intensity
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 10))
    for j in range(0, 360, angle_of_rotation):
            plt.plot((df[j]+ j*.01), alpha = .55, c = 'black')
    plt.xlabel('q A $^(-1)$')
    plt.ylabel('Intensity')
    plt.title("Waterfall Plot of Rotated 1D X-Ray Diffraction Images")
    plt.show()        
    
    return df

def run_nimfa_nmf(data, n_components):
    nmf = nimfa.Nmf(data, rank=n_components, max_iter=600, update='divergence', objective ='div', n_run = 30, track_error=True)
    nmf_fit = nmf()
    W = nmf_fit.basis()
    H = nmf_fit.coef()
    # print('Basis matrix:\n%s' % W.todense())
    # print('Mixture matrix:\n%s' % H.todense())
    er = nmf_fit.fit.tracker.get_error()

    print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))
    # Objective function value for each iteration
    print('Error tracking:\n%s' % nmf_fit.fit.tracker.get_error())

    sm = nmf_fit.summary()
    print('Sparseness Basis: %5.3f  Mixture: %5.3f' % (sm['sparseness'][0], sm['sparseness'][1]))
    print('Iterations: %d' % sm['n_iter'])
    #print('Target estimate:\n%s' % np.dot(W.todense(), H.todense()))
    
    return W, H, er

def run_HAC(Number_Clusters, data):
    """A program that will take in the type of scikitlearn clustering algorithm
    desired and the number of clusters as well as the data in a numpy array
    and output the associated clusters with the original data. This will make
    the 'latent' space from the clustering algorithms have more meaning"""
    
    # data_list = []
    # i = 0
    # while i < len(basis_data.T)-1:
    #     data_list.append(basis_data[:,i])
    #     i += 1
    # data = np.array(data_list)
    
    # print(data.shape)

    # Initialize and fit the Agglomerative Clustering model
    Make_Clusters = AgglomerativeClustering(n_clusters=Number_Clusters, compute_distances=True)
    y_kmeans = Make_Clusters.fit_predict(data)
    distances = Make_Clusters.distances_

    # Prepare the data for understanding
    Understanding_data = {
        "Cluster_Number": y_kmeans.tolist(),
        "Int_Angle": data.tolist()
    }               
    
    # Create an empty list to store the data
    data_list = []

    # Plot and analyze each cluster
    for q in range(Number_Clusters):
        plt.figure(figsize=(5, 5))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Agglomerative Clustering {q}')

        for z in range(len(data)):
            if Understanding_data["Cluster_Number"][z] == q:
                plt.plot(Understanding_data["Int_Angle"][z], label=f'Component {z}')
        
        #plt.legend()
        plt.show(block=False)
        plt.pause(0.1)

        if input("Do the identified components look like an isotropic scattering signal? (y/n) ") == 'y':
            for i in range(len(data)):
                if Understanding_data["Cluster_Number"][i] == q:
                    data_list.append(Understanding_data["Int_Angle"][i])

    return Understanding_data, data_list


from sklearn.preprocessing import StandardScaler

def run_nmf_and_agg_cluster(rotated_data, n_components, n_clusters):
    """A function that will run the NMF algorithm and then cluster with agglomerative clustering the components and returns the
    identified components for later PDF analysis."""
    
    #running nmf algorith and returning the basis, coefficient, and error matricies
    basis, coefficient, err = run_nimfa_nmf(rotated_data, n_components)
    
    print(basis.shape)
    
    #converting the basis matrix to a numpy array and data to input into the agglomerative clustering algorithm
    basis_np = np.array(basis)
    
    data = []
    i = 0
    while i < len(basis_np.T)-1:
        data.append(basis_np[:,i])
        i += 1
    data_np = np.array(data)
    
    #scaled_data = StandardScaler().fit_transform(data_np)
    
    #plotting error from the nmf run
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(err) + 1), err, marker='o')
    plt.title('Elbow Plot of NMF Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    plt.show()
    
    #running the returned basis matrix through the agglomerative clustering algorithm
    all_data, found_comps = run_HAC(n_clusters, data_np)
    
    return all_data, found_comps

def run_nmfac(Data, initialize_iter = 0, clusters = 5):
    """A function that will run the NMF algorithm and then cluster with agglomerative clustering the components and returns the 
        identified components for later PDF analysis. The function starts with a random initializer
        that will be used to initialize the NMF algorithm. The user can decide how many iterations the 
        initializer takes. Then it will go through the NMF algorithm and compare the beta divergences of all the
        initializations and select the one with the lowest. By default, there is no initializer and the NMF algorithm
        uses preset parameters to run the algorithm. """
        
        #NMF Parameter Values
        
    import warnings
    warnings.filterwarnings("ignore")
    
    
    init_params = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']
    solver_params = ['cd', 'mu']
    beta_loss_params = ['frobenius', 'kullback-leibler']
    tol_params = np.arange(0.00001, 0.01, 0.0001)
    max_iter_params = np.arange(100, 10000, 100)
    shuffle_params = [True, False]
        
    #When random initializers are not wanted
    if initialize_iter == 0:
        weights, components, beta = Run_NMF(Data, show = True)
        AggComponents = np.array(components).T
        AggClusters, found_compos = AggCluster(clusters, AggComponents)
        
        found_compos = np.array(found_compos)
        
    elif initialize_iter > 0:
        beta_div = []
        
        init_for_init = np.random.choice(init_params,initialize_iter)
        solver_for_init = np.random.choice(solver_params, initialize_iter)
        beta_loss_for_init = np.random.choice(beta_loss_params, initialize_iter)
        tol_for_init = np.random.choice(tol_params, initialize_iter)
        max_iter_for_init = np.random.choice(max_iter_params, initialize_iter)
        shuffle_for_init = np.random.choice(shuffle_params, initialize_iter)
        
        for i in range(initialize_iter):
            
            if beta_loss_for_init[i] == 'kullback-leibler':
                solver_for_init[i] = 'mu'
                
            print('The parameters selected for run' + str([i]) + ' are ' + str(init_for_init[i]) + ', ' + str(solver_for_init[i]) + ', ' + str(beta_loss_for_init[i]) + ', ' + str(tol_for_init[i]) + ', ' + str(max_iter_for_init[i]) + ', ' + str(shuffle_for_init[i]) + '. \n' )
            weights, components, beta = Run_NMF(Data, init = init_for_init[i], solver = solver_for_init[i], 
                                                beta_loss = beta_loss_for_init[i], itear = max_iter_for_init[i], show = False)
            
            beta_div.append(beta)
        beta_np = np.array(beta_div)
        good_init = np.argmin(beta_np)
        
        weights, components, beta = Run_NMF(Data, show = True, init = init_for_init[good_init], solver = solver_for_init[good_init], beta_loss=beta_loss_for_init[good_init],
                                             itear = max_iter_for_init[good_init])
        
        AggComponents = np.array(components).T
        AggClusters, found_compos = AggCluster(clusters, AggComponents)
        
        found_compos = np.array(found_compos)
            
        
    return found_compos

import numpy as np
from sklearn.decomposition import NMF
import logging
import warnings
import random
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_sklearn_nmf(data, max_components, max_iter=600, init='random', solver='cd', tol=1e-4, patience=5, randomize_init=True):
    """
    Run NMF using the sklearn library with flexible parameters and iterate over the number of components until error stabilizes.

    Parameters:
    - data: Input data matrix.
    - max_components: Maximum number of components to try.
    - max_iter: Maximum number of iterations (default: 600).
    - init: Initialization method (default: 'random').
    - solver: Solver to use (default: 'cd').
    - tol: Tolerance for error change to consider it stabilized (default: 1e-4).
    - patience: Number of runs to wait for error stabilization (default: 5).
    - randomize_init: Whether to randomly initialize parameters (default: True).

    Returns:
    - best_W: Best basis matrix.
    - best_H: Best coefficient matrix.
    - best_reconstruction_err: Best reconstruction error.
    - previous_errors: List of reconstruction errors from all runs.
    """
    
    warnings.filterwarnings("ignore")
    
    print('Starting NMF algorithm with the following parameters:\n')
    print('Max components: {}\nMax iterations: {}\nInit: {}\nSolver: {}\nTolerance: {}\nPatience: {}\nRandom'.format(max_components, max_iter, init, solver, tol, patience))


    best_W, best_H = None, None
    best_reconstruction_err = float('inf')
    best_n_components = 0
    previous_errors = []

    if randomize_init:
        init_options = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']
        solver_options = ['cd', 'mu']
        tol_options = [1e-4, 1e-5, 1e-6]

        for _ in range(10):  # Run 10 times with random initializers at 10 components each
            init = random.choice(init_options)
            solver = random.choice(solver_options)
            tol = random.choice(tol_options)
            
            model = NMF(n_components=10, max_iter=max_iter, init=init, solver=solver, tol=tol)
            W = model.fit_transform(data)
            H = model.components_
            reconstruction_err = model.reconstruction_err_

            logging.info('Random Init Run - Init: %s, Solver: %s, Tol: %e, Reconstruction error: %5.3f', init, solver, tol, reconstruction_err)
            
            if reconstruction_err < best_reconstruction_err:
                best_W, best_H = W, H
                best_reconstruction_err = reconstruction_err

        
        print('Random initializers completed\nContinuing with best run parameters that are init: {}, solver: {}, tol: {}'.format(init, solver, tol))
        
        # Continue with the best run's parameters
        init = model.init
        solver = model.solver
        tol = model.tol

    for n_components in range(1, max_components + 1):
        model = NMF(n_components=n_components, max_iter=max_iter, init=init, solver=solver, tol=tol)
        W = model.fit_transform(data)
        H = model.components_
        reconstruction_err = model.reconstruction_err_

        logging.info('Components: %d, Reconstruction error: %5.3f', n_components, reconstruction_err)
        
        print('Run with {} components has been completed'.format(n_components))

        if len(previous_errors) >= patience:
            if all(abs(previous_errors[-i] - reconstruction_err) < tol for i in range(1, patience + 1)):
                logging.info('Error has stabilized over the last %d runs.', patience)
                print('Error has stabilized over the last {} runs.'.format(patience))
                best_W, best_H = W, H
                best_reconstruction_err = reconstruction_err
                best_n_components = n_components
                break

        previous_errors.append(reconstruction_err)
        best_W, best_H = W, H
        best_reconstruction_err = reconstruction_err
        best_n_components = n_components
        
        
    # Create an elbow plot of the error values from all the runs
    plt.figure(figsize=(12, 6))
    plt.plot(previous_errors, marker='o', color='#5c146e', linestyle='-', linewidth=2, markersize=8, alpha=0.7)
    plt.title('Elbow Plot of NMF Reconstruction Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    plt.show()
        
    # Create a waterfall plot of the best W matrix
    plt.figure(figsize=(12, 6))
    colors = plt.cm.magma(np.linspace(0, 1, len(best_W[0])))

    for i in range(len(best_W[0])):
        plt.plot(best_W[:, i] + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])

    plt.title('Waterfall Plot of NMF Components')
    plt.xlabel('Data Points')
    plt.ylabel('Components')
    plt.grid(True)
    plt.show()
    
    print('The best number of components is {}'.format(best_n_components))

    return best_W, best_H, best_reconstruction_err

def cluster_results_basis(data, n_clusters):
    """
    Cluster the NMF results using agglomerative clustering and return the clusters.

    Parameters:
    - data: NMF results data matrix.
    - n_clusters: Number of clusters to create.

    Returns:
    - clusters: Cluster assignments for each data point.
    """
    from sklearn.cluster import AgglomerativeClustering

    cluster_data = np.array(data).T
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(cluster_data)
    
    #matching the clusters to the original data
    data_dict = {"Cluster_Number":[], "Component":[]}
    
    x=0
    while x < len(data[1]):
        data_dict["Cluster_Number"].append(clusters[x])
        data_dict["Component"].append(cluster_data[x])
        x = x+1
    
    q = 0
    while q < n_clusters:
        z = 0
        plt.figure(figsize=(5,5))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Agglomerative Clustering'+' ' +str(q))

        while z < len(data[1]):
            if data_dict["Cluster_Number"][z] == q:
                plt.plot(data_dict["Component"][z], label = 'Component'+str(z))
            z = z+1
        q = q+1
    
    return data_dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def cluster_results_weights(H_matrix, W_matrix, n_clusters):
    """
    Cluster the NMF results using agglomerative clustering and return the clusters.

    Parameters:
    - H_matrix: Coefficient matrix from NMF.
    - W_matrix: Basis matrix from NMF.
    - n_clusters: Number of clusters to create.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform agglomerative clustering on the H matrix
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(H_matrix)
    
    # Create a dictionary to store cluster assignments and associated components
    cluster_dict = {i: [] for i in range(n_clusters)}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format(cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def cluster_results_basis(W_matrix, n_clusters):
    """
    Cluster the NMF results using agglomerative clustering on the basis matrix and return the clusters.

    Parameters:
    - W_matrix: Basis matrix from NMF.
    - n_clusters: Number of clusters to create.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform agglomerative clustering on the W matrix
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(W_matrix.T)
    
    # Create a dictionary to store cluster assignments and associated components
    cluster_dict = {i: [] for i in range(n_clusters)}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format(cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def cluster_results_weights_dbscan(H_matrix, W_matrix, eps = 10, min_samples = 5):
    """
    Cluster the NMF results using DBSCAN and return the clusters.

    Parameters:
    - H_matrix: Coefficient matrix from NMF.
    - W_matrix: Basis matrix from NMF.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform DBSCAN clustering on the H matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(H_matrix)
    
    # Create a dictionary to store cluster assignments and associated components
    unique_clusters = set(clusters)
    cluster_dict = {cluster: [] for cluster in unique_clusters}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        if cluster == -1:
            colors = ['red'] * len(components)  # Use red for noise points
        else:
            colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format('Noise' if cluster == -1 else cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cluster_results_basis_dbscan(W_matrix, eps = 10, min_samples = 5):
    """
    Cluster the NMF results using DBSCAN on the basis matrix and return the clusters.

    Parameters:
    - W_matrix: Basis matrix from NMF.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform DBSCAN clustering on the W matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(W_matrix.T)
    
    # Create a dictionary to store cluster assignments and associated components
    unique_clusters = set(clusters)
    cluster_dict = {cluster: [] for cluster in unique_clusters}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        if cluster == -1:
            colors = ['red'] * len(components)  # Use red for noise points
        else:
            colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format('Noise' if cluster == -1 else cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

from sklearn.cluster import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt

def cluster_results_weights_hdbscan(H_matrix, W_matrix, min_cluster_size = 10, min_samples = 5):
    """
    Cluster the NMF results using HDBSCAN and return the clusters.

    Parameters:
    - H_matrix: Coefficient matrix from NMF.
    - W_matrix: Basis matrix from NMF.
    - min_cluster_size: The minimum size of clusters.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform HDBSCAN clustering on the H matrix
    clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clustering.fit_predict(H_matrix)
    
    # Create a dictionary to store cluster assignments and associated components
    unique_clusters = set(clusters)
    cluster_dict = {cluster: [] for cluster in unique_clusters}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        if cluster == -1:
            colors = ['red'] * len(components)  # Use red for noise points
        else:
            colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format('Noise' if cluster == -1 else cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

from sklearn.cluster import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt

def cluster_results_basis_hdbscan(W_matrix, min_cluster_size = 10, min_samples = 5):
    """
    Cluster the NMF results using HDBSCAN on the basis matrix and return the clusters.

    Parameters:
    - W_matrix: Basis matrix from NMF.
    - min_cluster_size: The minimum size of clusters.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - cluster_dict: Dictionary with cluster assignments and associated components.
    """
    # Perform HDBSCAN clustering on the W matrix
    clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clustering.fit_predict(W_matrix.T)
    
    # Create a dictionary to store cluster assignments and associated components
    unique_clusters = set(clusters)
    cluster_dict = {cluster: [] for cluster in unique_clusters}
    
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(W_matrix[:, i])
    
    # Plot the clusters
    for cluster, components in cluster_dict.items():
        plt.figure(figsize=(6, 6))
        if cluster == -1:
            colors = ['red'] * len(components)  # Use red for noise points
        else:
            colors = plt.cm.plasma(np.linspace(0, 1, len(components)))
        
        for i, component in enumerate(components):
            plt.plot(component + i * 0.1, label='Component {}'.format(i + 1), color=colors[i])
        
        plt.title('Cluster {} Components'.format('Noise' if cluster == -1 else cluster))
        plt.xlabel('Data Points')
        plt.ylabel('Components')
        plt.grid(True)
        plt.show()
    
    return cluster_dict

def run_sklearn_nmf_and_agg_cluster(data, max_components, max_iter=600, init='random', solver='cd', tol=1e-4, patience=5, n_clusters=5, cluster_matrix = 'W'):
    """
    Run NMF using the sklearn library with flexible parameters and cluster the results using agglomerative clustering.

    Parameters:
    - data: Input data matrix.
    - max_components: Maximum number of components to try.
    - max_iter: Maximum number of iterations (default: 600).
    - init: Initialization method (default: 'random').
    - solver: Solver to use (default: 'cd').
    - tol: Tolerance for error change to consider it stabilized (default: 1e-4).
    - patience: Number of runs to wait for error stabilization (default: 5).
    - n_clusters: Number of clusters to create.

    Returns:
    - best_W: Best basis matrix.
    - best_H: Best coefficient matrix.
    - best_reconstruction_err: Best reconstruction error.
    - clusters: Cluster assignments for the NMF results.
    """
    
    best_W, best_H, best_reconstruction_err = run_sklearn_nmf(data, max_components, max_iter, init, solver, tol, patience)
    if cluster_matrix == 'W':
        data_dict = cluster_results_basis(best_W, n_clusters)
    else:
        data_dict = cluster_results_weights(best_H, best_W, n_clusters)
    
    return best_W, best_H, best_reconstruction_err, data_dict

def run_sklearn_nmf_and_dbscan(data, max_components, max_iter=600, init='random', solver='cd', tol=1e-4, patience=5, eps = 10, min_samples = 5, cluster_matrix = 'W'):
    """
    Run NMF using the sklearn library with flexible parameters and cluster the results using dbscan clustering

    Parameters:
    - data: Input data matrix.
    - max_components: Maximum number of components to try.
    - max_iter: Maximum number of iterations (default: 600).
    - init: Initialization method (default: 'random').
    - solver: Solver to use (default: 'cd').
    - tol: Tolerance for error change to consider it stabilized (default: 1e-4).
    - patience: Number of runs to wait for error stabilization (default: 5).
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - best_W: Best basis matrix.
    - best_H: Best coefficient matrix.
    - best_reconstruction_err: Best reconstruction error.
    - clusters: Cluster assignments for the NMF results.
    """
    
    best_W, best_H, best_reconstruction_err = run_sklearn_nmf(data, max_components, max_iter, init, solver, tol, patience)
    if cluster_matrix == 'W':
        data_dict = cluster_results_basis_dbscan(best_W, eps, min_samples)
    else:
        data_dict = cluster_results_weights(best_W, best_H, eps, min_samples)
    
    return best_W, best_H, best_reconstruction_err, data_dict

def run_sklearn_nmf_and_hdbscan(data, max_components, max_iter=600, init='random', solver='cd', tol=1e-4, patience=5, min_cluster_size = 10, min_samples = 5, cluster_matrix = 'W'):
    """
    Run NMF using the sklearn library with flexible parameters and cluster the results using dbscan clustering

    Parameters:
    - data: Input data matrix.
    - max_components: Maximum number of components to try.
    - max_iter: Maximum number of iterations (default: 600).
    - init: Initialization method (default: 'random').
    - solver: Solver to use (default: 'cd').
    - tol: Tolerance for error change to consider it stabilized (default: 1e-4).
    - patience: Number of runs to wait for error stabilization (default: 5).
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - best_W: Best basis matrix.
    - best_H: Best coefficient matrix.
    - best_reconstruction_err: Best reconstruction error.
    - clusters: Cluster assignments for the NMF results.
    """
    
    best_W, best_H, best_reconstruction_err = run_sklearn_nmf(data, max_components, max_iter, init, solver, tol, patience)
    if cluster_matrix == 'W':
        data_dict = cluster_results_basis_dbscan(best_W, min_cluster_size , min_samples)
    else:
        data_dict = cluster_results_weights(best_W, best_H, min_cluster_size , min_samples)
    
    return best_W, best_H, best_reconstruction_err, data_dict