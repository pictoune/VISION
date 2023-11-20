import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from code.gradhorn import gradhorn
from code.middlebury import readflo
from code.utils import dataset, angular_error, endPoint_error, relative_norm_error
from skimage.color import rgb2gray


def compute_optical_flow(Ix, Iy, It, n, kernel=None, epsilon=1e-10):
    w = np.zeros((Ix.shape[0], Ix.shape[1], 2))
    n = int(n / 2)
    for i in range(n, Ix.shape[0] - n + 1):
        for j in range(n, Ix.shape[1] - n + 1):
            Ax = Ix[i - n : i + n + 1, j - n : j + n + 1]
            Ay = Iy[i - n : i + n + 1, j - n : j + n + 1]
            At = It[i - n : i + n + 1, j - n : j + n + 1]
            if kernel is not None:
                Ax *= kernel
                Ay *= kernel
                At *= kernel
            A = np.stack((Ax.flatten(), Ay.flatten()), axis=-1)
            B = -At.flatten()
            w[i, j, :] = np.linalg.pinv(A.T @ A + epsilon * np.eye(2)) @ A.T @ B
    return w


def lucas(I1, I2, n):
    Ix, Iy, It = gradhorn(I1, I2)
    return compute_optical_flow(Ix, Iy, It, n)


def gaussian_kernel(n, std, normalised=False):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= 2 * np.pi * (std**2)
    return gaussian2D


def lucas_gaussian(I1,I2,n,std):
    grad_x, grad_y, grad_t = gradhorn(I1, I2)
    flow_vectors = np.zeros((I1.shape[0], I1.shape[1], 2))
    kernel = gaussian_kernel(n, std)

    half_n = int(n / 2)
    epsilon = 1e-10  

    for i in range(half_n, I1.shape[0] - half_n):
        for j in range(half_n, I1.shape[1] - half_n):
            window_grad_x = grad_x[i - half_n:i + half_n, j - half_n:j + half_n]
            window_grad_y = grad_y[i - half_n:i + half_n, j - half_n:j + half_n]
            window_grad_t = grad_t[i - half_n:i + half_n, j - half_n:j + half_n]

            Ax = (kernel * window_grad_x).flatten()
            Ay = (kernel * window_grad_y).flatten()
            B = -(kernel * window_grad_t).flatten()

            A = np.vstack((Ax, Ay)).T
            ATA = A.T @ A

            flow_vectors[i, j, :] = np.linalg.inv(ATA + epsilon * np.eye(2)) @ A.T @ B

    return flow_vectors

def optimize_lucas_kanade(dataset_name, smallest_window, biggest_window, step):
    best_window_size = {}
    errors = {
        "mean": {"Angular error": [], "EndPoint error": [], "Relative norm error": []},
        "std": {"Angular error": [], "EndPoint error": [], "Relative norm error": []},
    }
    min_error = {}
    best_flow = {}

    for window_size in range(smallest_window, biggest_window, step):
        print(f"\rwindow_size : {window_size}/{biggest_window}", end="")

        I1 = plt.imread(f"data/{dataset_name}/{dataset['image 1'][dataset_name]}")
        I2 = plt.imread(f"data/{dataset_name}/{dataset['image 2'][dataset_name]}")

        if I1.ndim == 3:
            I1 = rgb2gray(I1)
        if I2.ndim == 3:
            I2 = rgb2gray(I2)

        estimated_flow = lucas(I1, I2, window_size)
        gt_flow = readflo(f"data/{dataset_name}/{dataset['groundtruth'][dataset_name]}")

        ang_error_img = angular_error(estimated_flow, gt_flow)
        mean_ang_error_img = np.mean(ang_error_img)
        epe_error_img = endPoint_error(estimated_flow, gt_flow)
        mean_epe_error_img = np.mean(epe_error_img)
        norm_error_img = relative_norm_error(estimated_flow, gt_flow)
        mean_norm_error_img = np.mean(norm_error_img)

        errors["mean"]["Angular error"].append(mean_ang_error_img)
        errors["std"]["Angular error"].append(np.std(ang_error_img))
        errors["mean"]["EndPoint error"].append(mean_epe_error_img)
        errors["std"]["EndPoint error"].append(np.std(epe_error_img))
        errors["mean"]["Relative norm error"].append(mean_norm_error_img)
        errors["std"]["Relative norm error"].append(np.std(norm_error_img))

        update_best_flow(
            window_size,
            smallest_window,
            mean_ang_error_img,
            mean_epe_error_img,
            mean_norm_error_img,
            estimated_flow,
            min_error,
            best_flow,
            best_window_size,
        )

    return min_error, best_flow, best_window_size, errors


def update_best_flow(
    window_size,
    smallest_window,
    mean_ang_error_img,
    mean_epe_error_img,
    mean_norm_error_img,
    estimated_flow,
    min_error,
    best_flow,
    best_window_size,
):
    if window_size == smallest_window:
        min_error["Angular error"] = mean_ang_error_img
        best_flow["Angular error"] = estimated_flow
        best_window_size["Angular error"] = window_size

        min_error["EndPoint error"] = mean_epe_error_img
        best_flow["EndPoint error"] = estimated_flow
        best_window_size["EndPoint error"] = window_size

        min_error["Relative norm error"] = np.abs(mean_norm_error_img)
        best_flow["Relative norm error"] = estimated_flow
        best_window_size["Relative norm error"] = window_size
    else:
        if mean_ang_error_img < min_error["Angular error"]:
            min_error["Angular error"] = mean_ang_error_img
            best_flow["Angular error"] = estimated_flow
            best_window_size["Angular error"] = window_size

        if mean_epe_error_img < min_error["EndPoint error"]:
            min_error["EndPoint error"] = mean_epe_error_img
            best_flow["EndPoint error"] = estimated_flow
            best_window_size["EndPoint error"] = window_size

        if np.abs(mean_norm_error_img) < min_error["Relative norm error"]:
            min_error["Relative norm error"] = np.abs(mean_norm_error_img)
            best_flow["Relative norm error"] = estimated_flow
            best_window_size["Relative norm error"] = window_size
