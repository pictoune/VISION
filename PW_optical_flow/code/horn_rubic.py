from matplotlib import pyplot as plt
import numpy as np
from middlebury import computeColor
from horn import horn
from middlebury import computeColor
from lucas import quiver

I1 = plt.imread("../data/rubic/rubic9.png")
I2 = plt.imread("../data/rubic/rubic10.png")

N = 50
fig = plt.figure()

for i,alpha in enumerate([0.05,0.1,0.2,0.8]):
    fig.add_subplot(1, 4, i+1)
    fig.tight_layout(pad=0.1)
    plt.imshow(computeColor(horn(I1,I2,alpha,N)))
    plt.title(r"$\alpha$ = %.2f" % alpha, fontsize=20)
plt.show()

quiver(horn(I1,I2,0.2,300),'Best flow as a vector field obtained with the Horn-Schunk method',20)
quiver(horn(I1,I2,0.2,300),'Best flow as a normalized vector field obtained with the Horn-Schunk method',40,norm=True)