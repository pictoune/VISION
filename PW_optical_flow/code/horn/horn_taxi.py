from matplotlib import pyplot as plt
from code.middlebury import computeColor
from code.horn.horn import horn
from code.utils import quiver

I1 = plt.imread("data/taxi/taxi9.png")
I2 = plt.imread("data/taxi/taxi10.png")

N = 700
fig = plt.figure()

alphas = [0.05, 0.1, 0.2, 0.8]
for i, alpha in enumerate(alphas):
    fig.add_subplot(1, 4, i+1)
    fig.tight_layout(pad=0.1)
    plt.imshow(computeColor(horn(I1, I2, alpha, N)))
    plt.title(r"$\alpha$ = %.2f" % alpha, fontsize=10)
plt.show()

best_flow = horn(I1, I2, 0.2, 300)
quiver(best_flow, 'Best flow as a vector field obtained with the Horn-Schunk method', 20)
quiver(best_flow, 'Best flow as a normalized vector field obtained with the Horn-Schunk method', 40)
