from matplotlib import pyplot as plt
from code.middlebury import computeColor
from code.lucas.lucas import lucas, lucas_gaussian
from code.utils import quiver

I1 = plt.imread("data/nasa/nasa9.png")
I2 = plt.imread("data/nasa/nasa10.png")

fig = plt.figure(figsize=(15, 4))
n_values = [2, 10, 24, 50]
lucas_gaussian_results = {}

for i, n in enumerate(n_values):
    fig.add_subplot(2, 4, i + 1)
    plt.imshow(computeColor(lucas(I1, I2, n)))
    plt.title(f"n = {n}", fontsize=16)

    if n not in lucas_gaussian_results:
        lucas_gaussian_results[n] = lucas_gaussian(I1, I2, n, 5)
    fig.add_subplot(2, 4, i + 5)
    plt.imshow(computeColor(lucas_gaussian_results[n]))
    plt.title(f"n = {n}", fontsize=16)

fig.tight_layout(pad=0.2)
plt.show()

best_flow = lucas_gaussian_results[24]
quiver(
    best_flow, "Best flow as a vector field obtained with the Lucas-Kanade method", 15
)
quiver(
    best_flow,
    "Best flow as a normalized vector field obtained with the Lucas-Kanade method",
    40,
)
