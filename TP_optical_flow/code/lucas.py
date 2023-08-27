import numpy as np
import matplotlib.pyplot as plt
from gradhorn import gradhorn
from scipy import signal
from middlebury import readflo
from utils import dataset, angular_error, endPoint_error, relative_norm_error
from skimage import io

def lucas(I1,I2,n):
    Ix,Iy,It = gradhorn(I1,I2)
    w = np.zeros((I1.shape[0], I1.shape[1],2))
    epsilon=0.0000000001

    n = int(n/2)
    for i in range(n,I1.shape[0]-n+1):
        for j in range(n,I1.shape[1]-n+1):
            Ax = Ix[i-n:i+n+1,j-n:j+n+1].flatten()
            Ay = Iy[i-n:i+n+1,j-n:j+n+1].flatten()
            A = np.dstack((Ax,Ay))
            A = A.reshape((A.shape[1],A.shape[2]))
            B = -It[i-n:i+n+1,j-n:j+n+1].flatten()
            w[i,j,:] = np.linalg.inv(A.T@A+epsilon*np.eye(2))@A.T@B
    return w

def gaussian_kernel(n, std, normalised=False):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def lucas_gaussian(I1,I2,n,std):
    Ix,Iy,It = gradhorn(I1,I2)
    w = np.zeros((I1.shape[0], I1.shape[1],2))
    kernel = gaussian_kernel(n,std)
    n = int(n/2)
    epsilon=0.0000000001
    for i in range(n,I1.shape[0]-n+1):
        for j in range(n,I1.shape[1]-n+1):
            Ax = (kernel*Ix[i-n:i+n,j-n:j+n]).flatten()
            Ay = (kernel*Iy[i-n:i+n,j-n:j+n]).flatten()
            A = np.dstack((Ax,Ay))
            A = A.reshape((A.shape[1],A.shape[2]))
            B = -(kernel*It[i-n:i+n,j-n:j+n]).flatten()
            w[i,j,:] = np.linalg.inv(A.T@A+epsilon*np.eye(2))@A.T@B
    return w

def optimize_lucas_kanade(dataset_name,smallest_window,biggest_window,step):
    def rgb2gray(rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    best_window_size = {}
    
    errors = {}
    errors["mean"] = {}
    errors["mean"]["Angular error"] = []
    errors["mean"]["EndPoint error"] = []
    errors["mean"]["Relative norm error"] = []
    
    errors["std"] = {}
    errors["std"]["Angular error"] = []
    errors["std"]["EndPoint error"] = []
    errors["std"]["Relative norm error"] = []
    
    min_error = {}
    
    best_flow = {}

    for window_size in range(smallest_window,biggest_window,step):
        print("\rwindow_size : " + str(window_size) + "/" + str(biggest_window),end='')

        I1 = plt.imread("../data/" + dataset_name + "/" + dataset["image 1"][dataset_name])
        I2 = plt.imread("../data/" + dataset_name + "/" + dataset["image 2"][dataset_name])        
        
        if(I1.ndim == 3):
            I1 = rgb2gray(I1)
        if(I2.ndim == 3):
            I2 = rgb2gray(I2)

        estimated_flow = lucas(I1,I2,window_size) 
        gt_flow = readflo('../data/' + dataset_name + '/' + dataset["groundtruth"][dataset_name])
        
        ang_error_img = angular_error(estimated_flow,gt_flow)
        mean_ang_error_img = np.mean(ang_error_img)
        
        errors["mean"]["Angular error"].append(mean_ang_error_img)
        errors["std"]["Angular error"].append(np.std(ang_error_img))
        
        epe_error_img = endPoint_error(estimated_flow,gt_flow)
        mean_epe_error_img = np.mean(epe_error_img)
        
        errors["mean"]["EndPoint error"].append(mean_epe_error_img)
        errors["std"]["EndPoint error"].append(np.std(epe_error_img))
        
        norm_error_img = relative_norm_error(estimated_flow,gt_flow)
        mean_norm_error_img = np.mean(norm_error_img)
        
        errors["mean"]["Relative norm error"].append(np.mean(norm_error_img))  
        errors["std"]["Relative norm error"].append(np.std(norm_error_img))  
        
        if(window_size == smallest_window):
            min_error["ang"] = mean_ang_error_img
            best_flow["ang"] = estimated_flow
            best_window_size["angular error"] = window_size
            
            min_error["EPE"] = mean_epe_error_img
            best_flow["EPE"] = estimated_flow
            best_window_size["EPE"] = window_size
            
            min_error["norm"] = np.abs(mean_norm_error_img)
            best_flow["norm"] = estimated_flow
            best_window_size["norm"] = window_size
        else:
            if(mean_ang_error_img < min_error["ang"]):
                min_error["ang"] = mean_ang_error_img
                best_flow["ang"] = estimated_flow
                best_window_size["angular error"] = window_size

            if(mean_epe_error_img < min_error["EPE"]):
                min_error["EPE"] = mean_epe_error_img
                best_flow["EPE"] = estimated_flow
                best_window_size["EPE"] = window_size

            if(np.abs(mean_norm_error_img) < min_error["norm"]):
                min_error["norm"] = np.abs(mean_norm_error_img)
                best_flow["norm"] = estimated_flow
                best_window_size["norm"] = window_size
    
    return min_error,best_flow,best_window_size,errors