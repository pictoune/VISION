import cv2
import numpy as np
from scipy import signal


class ROI:
    """
    Represents a Region of Interest (ROI) in an image.

    Methods:
        __init__(): Initializes the ROI object.
        define_ROI(event, x, y, flags, param): Defines the ROI based on mouse events.

    Args:
        event: The mouse event type.
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Additional flags for the mouse event.
        param: Additional parameters for the mouse event.
    """
    def __init__(self):
        self.r = self.c = self.w = self.h = 0
        self.roi_defined = False

    def define_ROI(self, event, x, y, flags, param):
        # if the left mouse button was clicked,
        # record the starting ROI coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.r, self.c = x, y
            self.roi_defined = False
        # if the left mouse button was released,
        # record the ROI coordinates and dimensions
        elif event == cv2.EVENT_LBUTTONUP:
            r2, c2 = x, y
            self.h = abs(r2 - self.r)
            self.w = abs(c2 - self.c)
            self.r = min(self.r, r2)
            self.c = min(self.c, c2)
            self.roi_defined = True


def get_angle(vec):
    """
    Returns the angle in degrees between the positive x-axis and the vector (i, j).

    Args:
        vec (tuple): A tuple representing the vector (i, j).

    Returns:
        float: The angle in degrees between the positive x-axis and the vector.
    """
    i, j = vec

    return np.rad2deg(np.arctan2(j, i))


def compute_grad(I):
    """
    Computes the gradient magnitude and orientation of an image.

    Args:
        I: The input image.

    Returns:
        mod: The gradient magnitude of the image.
        ori: The gradient orientation of the image.
    """
    hy = np.array([[1 / 2, 1, 1 / 2]])
    hx = np.array([[-1 / 2, 0, 1 / 2]])

    Ix = signal.convolve2d(I, hy.T @ hx, mode="same")
    Iy = signal.convolve2d(I, hx.T @ hy, mode="same")

    mod = np.sqrt(Ix**2 + Iy**2)
    ori = np.arctan2(Iy, Ix)

    return mod, ori


# In utils.py

def process_roi(frame, myROI, clone):
    """
    Process the ROI by drawing it on the frame if defined, or resetting the frame otherwise.

    Args:
        frame: The current video frame.
        myROI: An instance of the ROI class.
        clone: A clone of the original frame for reset purposes.

    Returns:
        Processed frame.
    """
    processed_frame = frame.copy()
    if myROI.roi_defined:
        # Draw a rectangle around the ROI
        cv2.rectangle(processed_frame, (myROI.r, myROI.c),
                      (myROI.r + myROI.h, myROI.c + myROI.w),
                      (0, 255, 0), 2)
    else:
        # Reset the frame to the original
        processed_frame = clone.copy()
    return processed_frame
