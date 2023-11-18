import numpy as np
import matplotlib.pyplot as plt
from gradhorn import gradhorn
from scipy.signal import convolve2d
from utils import dataset, endPoint_error, relative_norm_error, angular_error
import middlebury

def horn(I1,I2,alpha,N):
    Ix,Iy,It = gradhorn(I1,I2)
    u=np.zeros(I1.shape)
    v=np.zeros(I1.shape)
    A = 1/6*np.asarray([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])
    for k in range(N):
        u_bar = convolve2d(u,A,mode='same')
        v_bar = convolve2d(v,A,mode='same')
        u = u_bar-Ix*(Ix*u+Iy*v+It)/(alpha+Ix*Ix+Iy*Iy)
        v = v_bar-Iy*(Ix*u+Iy*v+It)/(alpha+Ix*Ix+Iy*Iy)
    return np.dstack((u,v))

def optimize_horn_schunck(dataset_name,smallest_alpha,biggest_alpha,step,nb_iter=200):
    def rgb2gray(rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    
    best_alpha = {}
    
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
    
    for alpha in np.arange(smallest_alpha,biggest_alpha,step):
        print("\ralpha : " + str(round(alpha,2)) + "/" + str(round(biggest_alpha,2)),end='')
        
        I1 = plt.imread("../data/" + dataset_name + "/" + dataset["image 1"][dataset_name])
        I2 = plt.imread("../data/" + dataset_name + "/" + dataset["image 2"][dataset_name])
        
        if(I1.ndim == 3):
            I1 = rgb2gray(I1)
        if(I2.ndim == 3):
            I2 = rgb2gray(I2)
        
        estimated_flow = horn(I1,I2,alpha,nb_iter)
        
        gt_flow = middlebury.readflo('../data/' + dataset_name + '/' + dataset["groundtruth"][dataset_name])
        
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
        
        if(alpha == smallest_alpha):
            min_error["ang"] = mean_ang_error_img
            best_flow["ang"] = estimated_flow
            best_alpha["angular error"] = alpha
            
            min_error["EPE"] = mean_epe_error_img
            best_flow["EPE"] = estimated_flow
            best_alpha["EPE"] = alpha
            
            min_error["norm"] = np.abs(mean_norm_error_img)
            best_flow["norm"] = estimated_flow
            best_alpha["norm"] = alpha
            

        if(mean_ang_error_img < min_error["ang"]):
            min_error["ang"] = mean_ang_error_img
            best_flow["ang"] = estimated_flow
            best_alpha["angular error"] = alpha
            
        if(mean_epe_error_img < min_error["EPE"]):
            min_error["EPE"] = mean_epe_error_img
            best_flow["EPE"] = estimated_flow
            best_alpha["EPE"] = alpha
        
        if(np.abs(mean_norm_error_img) < min_error["norm"]):
            min_error["norm"] = np.abs(mean_norm_error_img)
            best_flow["norm"] = estimated_flow
            best_alpha["norm"] = alpha
    
    return min_error,best_flow,best_alpha,errors