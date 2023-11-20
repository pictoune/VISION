from matplotlib import pyplot as plt
from code.middlebury import computeColor
from code.lucas.lucas import lucas, lucas_gaussian
from code.utils import quiver


def plot_flow(method, title, position):
    plt.subplot(2, 4, position)
    plt.imshow(computeColor(method))
    plt.title(title, fontsize=16)


I1 = plt.imread("data/taxi/taxi9.png")
I2 = plt.imread("data/taxi/taxi10.png")

fig = plt.figure(figsize=(15, 4))

for i, n in enumerate([2, 6, 10, 16]):
    plot_flow(lucas(I1, I2, n), f"n = {n}", i + 1)
    plot_flow(lucas_gaussian(I1, I2, n, 5), f"n = {n}", i + 5)

fig.tight_layout(pad=0.2)
plt.show()

quiver(
    lucas_gaussian(I1, I2, 24, 5),
    "Best flow as a vector field obtained with the Lucas-Kanade method",
    40,
)
quiver(
    lucas_gaussian(I1, I2, 24, 5),
    "Best flow as a normalized vector field obtained with the Lucas-Kanade method",
    40,
)
