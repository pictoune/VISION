import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray

from code.gradhorn import gradhorn
from code.utils import dataset, endPoint_error, relative_norm_error, angular_error
import code.middlebury as middlebury

def horn(I1, I2, alpha, N):
    """
    Compute the optical flow using the Horn-Schunck algorithm.

    Args:
        I1 (ndarray): The first input image.
        I2 (ndarray): The second input image.
        alpha (float): The regularization parameter.
        N (int): The number of iterations.

    Returns:
        ndarray: The computed optical flow as a stack of u and v components.
    """
    Ix, Iy, It = gradhorn(I1, I2)
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    A = 1/6*np.asarray([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])

    for _ in range(N):
        u_bar = convolve2d(u,A,mode='same')
        v_bar = convolve2d(v,A,mode='same')
        denom = alpha + Ix**2 + Iy**2
        u = u_bar - Ix * (Ix * u + Iy * v + It) / denom
        v = v_bar - Iy * (Ix * u + Iy * v + It) / denom
    
    return np.dstack((u, v))

def optimize_horn_schunck(dataset_name, smallest_alpha, biggest_alpha, step, nb_iter=200):
    """
    Optimizes the Horn-Schunck optical flow algorithm for a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        smallest_alpha (float): The smallest value of alpha to iterate over.
        biggest_alpha (float): The largest value of alpha to iterate over.
        step (float): The step size for alpha iteration.
        nb_iter (int, optional): The number of iterations for the algorithm. Defaults to 200.

    Returns:
        tuple: A tuple (floats) containing the minimum error, best flow, best alpha, and error statistics.
    """
    best_alpha = {}
    errors = {"mean": {"Angular error": [], "EndPoint error": [], "Relative norm error": []},
              "std": {"Angular error": [], "EndPoint error": [], "Relative norm error": []}}
    min_error = {}
    best_flow = {}

    for alpha in np.arange(smallest_alpha, biggest_alpha, step):
        print(f"\ralpha : {round(alpha, 2)}/{round(biggest_alpha, 2)}", end='')

        I1 = plt.imread(f"data/{dataset_name}/{dataset['image 1'][dataset_name]}")
        I2 = plt.imread(f"data/{dataset_name}/{dataset['image 2'][dataset_name]}")

        I1 = rgb2gray(I1) if I1.ndim == 3 else I1
        I2 = rgb2gray(I2) if I2.ndim == 3 else I2

        estimated_flow = horn(I1, I2, alpha, nb_iter)
        gt_flow = middlebury.readflo(f"data/{dataset_name}/{dataset['groundtruth'][dataset_name]}")

        for error_type, error_func in [("Angular error", angular_error), ("EndPoint error", endPoint_error), ("Relative norm error", relative_norm_error)]:
            error_img = error_func(estimated_flow, gt_flow)
            mean_error_img = np.mean(error_img)
            errors["mean"][error_type].append(mean_error_img)
            errors["std"][error_type].append(np.std(error_img))

            if alpha == smallest_alpha or mean_error_img < min_error.get(error_type, float('inf')):
                min_error[error_type] = mean_error_img
                best_flow[error_type] = estimated_flow
                best_alpha[error_type] = alpha

    return min_error, best_flow, best_alpha, errors
