from PIL import Image
import pyFAI, fabio
from pyFAI.gui import jupyter
import pyFAI
import os
import matplotlib.pyplot as plt
import SimDAT2D as sim

#making the tiff image
data = sim.combine_image(sim.create_anisotropic(20, 20, 200, 100), sim.create_iso_no_input(.4, .5e-10, cmap = 'magma'))
noisy_data = sim.generate_noisemap(data)


import IsoDAT2D as iso
import SimDAT2D as sim
import dask.array as da
import dask.dataframe as dd
import masking
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def image_rotation(image, angle, show = False):
    """
    This function rotates the combined image by a user specified angle amount, if the angle specified is 1, the result is that the combined image is rotated by one degree.
    
    Parameters:
        image (2D array): The image of the combined spots and calibration.
        angle_of_rotation (int): The angle of rotation.
        """
    pil_format = Image.fromarray(image)
    rotated_image = pil_format.rotate(angle)
    rotated_image = np.array(rotated_image)
    
    if show == True:
        #display the rotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_image, cmap='viridis')
        plt.title("Rotated Image")
        plt.show()
    return rotated_image

#Create a function that takes the combined image and integrates it using the azimuthal integrator and displays the 1D image
def integrate_image(combined_image, distance, wavelength, resolution = 2500, mask = None, show = False, radial_range = None):
    """
    This function integrates the combined image using the azimuthal integrator and displays the 1D image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
    """
    #initialize the azimuthal integrator
    
     # Initialize the detector
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()
    
    
    ai = AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)
    
    #integrate the combined image using the azimuthal integrator
    q, I = ai.integrate1d(combined_image, resolution, radial_range = radial_range, unit = 'q_A^-1', mask = mask)
    
    if show == True:
        #plot the 1D image
        plt.figure(figsize=(10, 10))
        plt.plot(q, I)
        plt.title("1D X-Ray Diffraction Image")
        plt.show()
    
    return q, I
    

def rotate_and_integrate(combined_image, angle_of_rotation, distance, wavelength, resolution = 1000, mask = None):
    """
    This function takes the combined image, the mask, the distance, the wavelength, and the resolution of integration, and rotates the combined image by a user specified angle amount, if the angle specified is 1, the result will be 360 integrations of the combined image, each integration will be rotated by 1 degree.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        angle_of_rotation (int): The angle of rotation.
        distance (float): The distance from the detector to the sample.
        wavelength (float): The wavelength of the x-rays.
        resolution (int): The resolution of the integration.
        mask (2D array): The mask to use for the integration.
    """
    
    import pandas as pd 
    
    #create a dataframe to store the 1D integrations
    df = pd.DataFrame()
    
    #create a loop that rotates the combined image by the user specified angle amount and integrates the image
    for i in range(0, 360, angle_of_rotation):
        #rotate the mask for the combined image
        rotated_image = image_rotation(combined_image, i);
    
        
        #integrate the rotated image
        q, I = integrate_image(rotated_image, distance, wavelength, resolution, mask, show = False, radial_range = (0, 25));
        
        #add the 1D integration to the dataframe
        df[i] = I
        
        #create a waterfall plot of the 1D integrations, where each dataset is moved up on the y axis by a multiple of .5
    plt.figure(figsize=(10, 10))
    for j in range(0, 360, angle_of_rotation):
            plt.plot(q, (df[j]+ j*.01), alpha = .55, c = 'black')
    plt.xlabel('q A $^(-1)$')
    plt.ylabel('Intensity')
    plt.title("Waterfall Plot of Rotated 1D X-Ray Diffraction Images")
    plt.show()        
    return q, df

def make_masks(array, slices, offset = 7, width=.5):
    masks = []
    for i in slices:
        masks.append(masking.generate_mask_slices(array, width, i, offset = offset))
        print('Mask with {} slices created'.format(i))
    return masks

def generate_mask_slices(array, width, num_slices, offset = 5):
    
    ''' Returns a mask with multiple slices of the chi array left unmasked to be used for integration.
    
    Keyword arguments:
    chi_array -- chi array
    width -- width of the slice in degrees
    num_slices -- number of slices
    offset -- offset between slices in degrees
    plot -- if True, plots the mask (default False)
    
    '''
    mask_list = []
    
    # Create masks for the positive values
    for i in range(num_slices):
        start = i * (width + offset)
        end = start + width
        mask_list.append(ma.masked_inside(array, start, end))

    # Create masks for the negative values
    for i in range(num_slices):
        start = - (i + 1) * (width + offset)
        end = start - width
        mask_list.append(ma.masked_inside(array, start, end))
    
    #add all genrated masks together
    
    print(mask_list)

    combined_mask = mask_list[0]
    for mask in mask_list[1:]:
        combined_mask += mask
        
    inverted_mask = ~combined_mask.mask 
    plt.figure()
    plt.imshow(~combined_mask.mask)
    
    return inverted_mask


import IsoDAT2D as iso
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import SimDAT2D as sim
import os
import IsoDAT2D as iso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.decomposition import NMF
import dask
import glob as glob
from tifffile import imread, imshow
import warnings
import dask
import pickle 
from dask.distributed import Client, as_completed, get_worker
import pandas as pd

#data = pd.read_csv('integrations_more.csv')


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


    # calcs = []
    
    # for i in range(1, Real_Data.shape[1]):
    #     calcs.append(attempt(Real_Data, Real_Data.shape[1], i, In, Solve, Beta, It))
    #     print('The beta-divergence for', i, 'components is', calcs[i-1], '%\n')
    #     print(f'Completed {i} out of {Real_Data.shape[1]}')
        
    # calcs = np.array(calcs)
    # min_Q = np.min(calcs)
    # noc = np.where(calcs == min_Q)
    # noc_2 = noc[0]
    # number_of_components = noc_2[0] +1

    Divergence, compos, NMF_Data_2 = attempt2(Real_Data, Real_Data.shape[1], len(Real_Data.columns), In, Solve, Beta)
            
    
    print('The beta-divergence is: ', Divergence, '%\n','The final number of components used were',len(Real_Data.columns), '\n') 
    
    
    m = pd.DataFrame(compos)
    m = m.T
    
    if show == True:
        plt.figure(figsize = (5,5))
        colors = plt.cm.magma(np.linspace(0,1, len(Real_Data.columns)))
        i = 0
        while i < len(Real_Data.columns):
            plt.plot(m[i], c = colors[i], alpha = 0.7)
            i = i+1
    
    return m,NMF_Data_2, Divergence

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
    Understanding_data = {"Cluster_Number":[], "Int_Angle":[]}
    while x < len(data):
        Understanding_data["Cluster_Number"].append(y_kmeans[x])
        Understanding_data["Int_Angle"].append(data[x])
        x = x + 1

    # Create a dictionary to store data points for each cluster
    cluster_data = {}
    for q in range(Number_Clusters):
        cluster_data[q] = []

    for i in range(len(data)):
        cluster_num = Understanding_data["Cluster_Number"][i]
        data_point = Understanding_data["Int_Angle"][i]
        cluster_data[cluster_num].append(data_point)

    return Understanding_data, cluster_data


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

def run_nmfac(Data, clusters = 5):
    """A function that will run the NMF algorithm and then cluster with agglomerative clustering the components and returns the 
        identified components for later PDF analysis. The function starts with a random initializer
        that will be used to initialize the NMF algorithm. The user can decide how many iterations the 
        initializer takes. Then it will go through the NMF algorithm and compare the beta divergences of all the
        initializations and select the one with the lowest. By default, there is no initializer and the NMF algorithm
        uses preset parameters to run the algorithm. """
        
        #NMF Parameter Values
        
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")
    
    
    init_params = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']
    solver_params = ['cd', 'mu']
    beta_loss_params = ['frobenius', 'kullback-leibler']
    tol_params = np.arange(0.00001, 0.01, 0.0001)
    max_iter_params = np.arange(100, 10000, 100)
    shuffle_params = [True, False]
    
    percentage =  100 # Change this to the desired percentage

    # Calculate the number of datasets based on the percentage
    num_datasets = int(len(Data.columns) * (percentage / 100))

    # Randomly select the dataset indices
    random_indices = np.random.choice(len(Data.columns), size=num_datasets, replace=False)
    
    transposed_data = Data.T

    # Select the randomly chosen datasets
    random_datasets_transposed = transposed_data.iloc[random_indices]
    
    random_datasets = random_datasets_transposed.T

    # Print the randomly selected datasets
    print(random_datasets)
        
    weights, components, beta = Run_NMF(random_datasets, show = True)
    AggComponents = np.array(components).T
    Dictionary, found_compos = AggCluster(clusters, AggComponents)
    
    found_compos = np.array(found_compos)
    
        
        
    found_compos = np.array(found_compos)
            
    return Dictionary, found_compos



array, ai_pe = masking.make_chi_array(noisy_data, .4, .5e-10)
#pure = sim.create_isotropic(.4, .4e-10, cmap = 'magma')
masks = make_masks(array, [1,2,3, 4, 5])
mask_one = masks[0]
mask_two = masks[1]
mask_three = masks[2]
mask_four = masks[3]
mask_five = masks[4]
#more_masks = make_masks(array, [6, 7, 8, 9, 10])
#mask_six = more_masks[0]
#mask_seven = more_masks[1]
#mask_eight = more_masks[2]
#mask_nine = more_masks[3]
#mask_ten = more_masks[4]
#more_mmasks = make_masks(array, [3,5], offset = 5, width =.75,)
#mask_eleven = more_mmasks[0]
#mask_twelve= more_mmasks[1]


q, ints = rotate_and_integraten(noisy_data, 1, .4, .5e-10, resolution = 200, mask = mask_one)
q, ints1 = rotate_and_integrate(noisy_data, 1, .4, .5e-10, resolution = 200, mask = mask_two)
q, ints2 = rotate_and_integrate(noisy_data, 1, .4, .5e-10, resolution = 200, mask = mask_three)
q, ints3 = rotate_and_integrate(noisy_data, 1, .4, .5e-10, resolution = 200, mask = mask_four)
q, ints4 = rotate_and_integrate(noisy_data, 1, .4, .5e-10, resolution = 200, mask = mask_five)
#q, ints5 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_six)
#q, ints6 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_seven)
#q, ints7 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_eight)
#q, ints8 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_nine)
#q, ints9 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_ten)
#q, ints10 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_eleven)
#q, ints11 = rotate_and_integrate(sc_img, 1, .254, .1667e-10, resolution = 1500, mask = mask_twelve)

column_mapping = {}
for i in range(361, 721):
    column_mapping[(i - 361)] = (i)
ints1.rename(columns=column_mapping, inplace=True) 
   
column_mapping = {}
for i in range(722, 1082):
    column_mapping[(i - 722)] = (i)
print(column_mapping)
ints2.rename(columns=column_mapping, inplace=True) 

column_mapping = {}
for i in range(1083, 1443):
    column_mapping[(i - 1083)] = (i)
print(column_mapping)
ints3.rename(columns=column_mapping, inplace=True)  
  
column_mapping = {}
for i in range(1444, 1804):
    column_mapping[(i - 1444)] = (i)
ints4.rename(columns=column_mapping, inplace=True)  

#column_mapping = {}
#for i in range(1805, 2165):
#    column_mapping[(i - 1805)] = (i)
#ints5.rename(columns=column_mapping, inplace=True) 
   
#column_mapping = {}
#for i in range(2166, 2526):
#    column_mapping[(i - 2166)] = (i)
#print(column_mapping)
#ints6.rename(columns=column_mapping, inplace=True) 
#
#column_mapping = {}
#for i in range(2527, 2887):
#    column_mapping[(i - 2527)] = (i)
#print(column_mapping)
#ints7.rename(columns=column_mapping, inplace=True)    
#
#column_mapping = {}
#for i in range(2888, 3248):
#    column_mapping[(i - 2888)] = (i)
#ints8.rename(columns=column_mapping, inplace=True) 
 
#for i in range(3249, 3609):
#    column_mapping[(i - 3249)] = (i) 
#ints9.rename(columns=column_mapping, inplace=True)  

#for i in range(3610, 3970):
#    column_mapping[(i - 3610)] = (i) 
#ints10.rename(columns=column_mapping, inplace=True)

#for i in range(3971, 4331):
#    column_mapping[(i - 3971)] = (i) 
#ints11.rename(columns=column_mapping, inplace=True)

data_m = pd.concat([ints, ints1, ints2, ints3, ints4], axis = 1)
data_m = data_m.clip(lower=0)


my_dict2, my_comps2 = run_nmfac(data_m, clusters = 10)

cluster_groups2 = {}
for cluster_num in set(my_dict2['Cluster_Number']):
    cluster_groups2[cluster_num] = []

# Group the data based on cluster numbers
for cluster_num, int_angle in zip(my_dict2['Cluster_Number'], my_dict2['Int_Angle']):
    cluster_groups2[cluster_num].append(int_angle)
    
    
import numpy as np
np.save('noisy_data_resolution_two.npy', cluster_groups2)
