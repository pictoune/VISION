from utils import compute_grad
import cv2
import numpy as np


def get_orientation_mask(orientation, magnitude, threshold=20):
    """
    Generates a mask based on orientation and magnitude of gradients.

    Args:
        orientation (np.array): The gradient orientation of the image.
        magnitude (np.array): The gradient magnitude of the image.
        threshold (float): Threshold for gradient magnitude.

    Returns:
        np.array: Mask where the orientation is valid based on the magnitude threshold.
    """
    mask = magnitude > threshold
    valid_orientation = np.where(mask, orientation, 0)
    return valid_orientation


def apply_hough_transform(orientation, magnitude, R_table, discretization=20):
    """
    Applies the Hough transform for object tracking.

    Args:
        orientation (np.array): The gradient orientation of the image.
        magnitude (np.array): The gradient magnitude of the image.
        R_table (dict): The reference table for Hough transform.
        discretization (int): Number of bins for orientation discretization.

    Returns:
        np.array: The result of the Hough transform.
    """
    H = np.zeros(orientation.shape)
    discret = 360 / discretization

    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if magnitude[i, j] > 20:
                for ind in R_table[int(orientation[i, j] / discret)]:
                    x, y = int(i + ind[0]), int(j + ind[1])
                    if 0 <= x < H.shape[0] and 0 <= y < H.shape[1]:
                        H[x, y] += 1

    return H


def calculate_histogram_hsv(roi, mask_range=((0., 30., 20.), (180., 255., 235.))):
    """
    Calculates the HSV histogram of a region of interest (ROI).

    Args:
        roi (np.array): The region of interest in the image.
        mask_range (tuple): The lower and upper range for the mask in HSV space.

    Returns:
        np.array: The normalized HSV histogram of the ROI.
    """
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array(mask_range[0]), np.array(mask_range[1]))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist
