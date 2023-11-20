from matplotlib import pyplot as plt
from code.middlebury import computeColor
from code.lucas.lucas import lucas, lucas_gaussian
from code.utils import quiver

I1 = plt.imread("data/rubic/rubic9.png")
I2 = plt.imread("data/rubic/rubic10.png")

fig = plt.figure(figsize=(15, 4))

for i, n in enumerate([2, 6, 10, 24]):
    fig.add_subplot(2, 4, i + 1)
    plt.imshow(computeColor(lucas(I1, I2, n)))
    plt.title(f"n = {n}", fontsize=16)
    fig.add_subplot(2, 4, i + 5)
    plt.imshow(computeColor(lucas_gaussian(I1, I2, n, 5)))
    plt.title(f"n = {n}", fontsize=16)

fig.tight_layout(pad=0.2)
plt.show()

best_flow = lucas_gaussian(I1, I2, 24, 5)
quiver(
    best_flow, "Best flow as a vector field obtained with the Lucas-Kanade method", 25
)
quiver(
    best_flow,
    "Best flow as a normalized vector field obtained with the Lucas-Kanade method",
    40,
)
