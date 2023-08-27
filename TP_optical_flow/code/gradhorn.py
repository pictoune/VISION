import numpy as np
import os
import scipy
import matplotlib.pyplot as plt


def gradhorn(I1, I2):
    filter_x = 0.25*np.asarray([[0,0,0],[0,-1,1],[0,-1,1]])
    filter_y = 0.25*np.asarray([[0,0,0],[0,-1,-1],[0,1,1]])
    filter_t = 0.25*np.asarray([[0,0,0],[0,1,1],[0,1,1]])
    Ix = scipy.signal.convolve2d(I1,filter_x,mode='same')+scipy.signal.convolve2d(I2,filter_x,mode='same')
    Iy = scipy.signal.convolve2d(I1,filter_y,mode='same')+scipy.signal.convolve2d(I2,filter_y,mode='same')
    It = scipy.signal.convolve2d(I2,filter_t,mode='same')-scipy.signal.convolve2d(I1,filter_t,mode='same')
    return Ix,Iy,It

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__ == "__main__":
    I1 = plt.imread("../data/yosemite/yos9.png")
    I2 = plt.imread("../data/yosemite/yos10.png")
    Ix,Iy,It = gradhorn(I1,I2)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(Ix)
    plt.title('I_x')
    fig.add_subplot(1, 3, 2)
    plt.imshow(Iy)
    plt.title('I_y')
    fig.add_subplot(1, 3, 3)
    plt.imshow(It)
    plt.title('I_t')
    plt.show()