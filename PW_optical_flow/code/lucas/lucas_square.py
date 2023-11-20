import matplotlib.pyplot as plt
from code.lucas.lucas import optimize_lucas_kanade
import code.middlebury as middlebury
from code.utils import (
    angular_error,
    endPoint_error,
    relative_norm_error,
    display_results,
    dataset,
)


def plot_histogram(errors, title):
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=10)
    plt.title(title)
    plt.show()


smallest_window = 3
biggest_window = 100
step = 1

dataset_name = "square"
involved_parameter = "window's size"

min_error, best_flow, best_window_size, errors = optimize_lucas_kanade(
    dataset_name, smallest_window, biggest_window, step
)

gt_flow = middlebury.readflo(
    f"data/{dataset_name}/" + dataset["groundtruth"][dataset_name]
)

ang_errors = angular_error(gt_flow, best_flow["Angular error"])
plot_histogram(
    ang_errors,
    "Histogram of the errors of the best flow obtained after an optimization based on angular error",
)

epe_errors = endPoint_error(gt_flow, best_flow["EndPoint error"])
plot_histogram(
    epe_errors,
    "Histogram of the errors of the best flow obtained after an optimization based on EndPoint Error",
)

norm_errors = relative_norm_error(gt_flow, best_flow["Relative norm error"])
plot_histogram(
    norm_errors,
    "Histogram of the errors of the best flow obtained after an optimization based on relative norm error",
)

display_results(
    smallest_window,
    biggest_window,
    step,
    best_flow,
    errors,
    involved_parameter,
    best_window_size,
    dataset_name,
)
