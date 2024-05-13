from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename).astype("float64") #returns np array of floats
    mean_x = np.mean(x, axis=0)
    centered_x = x - mean_x
    return centered_x

def get_covariance(dataset):
    # Your implementation goes here!
    n = len(dataset)
    return np.dot(np.transpose(dataset), dataset) / (n - 1)

def get_eig(S, m):
    # Your implementation goes here!
    eigvalues, eigvectors = eigh(S, subset_by_index = [len(S) - m,len(S) - 1]) #where m is largest eigenvalues/vector
    eigvalues = np.flip(eigvalues) #numpy.flip(inputarray)
    eigvectors = np.flip(eigvectors, axis=1)
    return np.diag(eigvalues), eigvectors

def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigvalues = eigh(S, eigvals_only= True) #compute all eigenval of the co matrix
    eigvalues = np.flip(eigvalues) #sorting
    tot_variance = np.sum(eigvalues) 
    cumul_variance = 0
    m = 0 #number of eigenvalu

    for eigvalues in sorted(eigh(S, eigvals_only = True), reverse =True):
        cumul_variance += eigvalues
        m += 1
        if cumul_variance / tot_variance >= prop: #check ratio 
            break

    eigvalues, eigvectors = eigh(S, subset_by_index = [len(S) - m,len(S) - 1])
    eigvalues = np.flip(eigvalues) #sorting desc 
    eigvectors = np.flip(eigvectors, axis = 1)

    return np.diag(eigvalues), eigvectors #convert sorted eigenval to diag matrix

def project_image(image, U):
    # Your implementation goes here!
    tr_im = np.dot(np.transpose(U), image)
    return np.dot(U, tr_im)

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency

    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2) #
    original_reshaped = orig.reshape(64, 64)
    proj_reshaped = proj.reshape(64, 64)
    
    img_1 = ax1.imshow(original_reshaped, aspect = "equal")
    ax1.set_title("Original")
    ax1.axis("off")  #axes tick hidden
    plt.colorbar(img_1, ax = ax1)
    
    img_2 = ax2.imshow(proj_reshaped, aspect = "equal")
    ax2.set_title("Projection")
    ax2.axis("off")  
    plt.colorbar(img_2, ax = ax2)
    
    return fig, ax1, ax2


