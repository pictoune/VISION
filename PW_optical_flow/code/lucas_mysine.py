import matplotlib.pyplot as plt
from lucas import optimize_lucas_kanade
from utils import angular_error, endPoint_error, relative_norm_error, display_results, dataset
import middlebury

smallest_window = 3
biggest_window = 100
step = 1

dataset_name = "mysine"
involved_parameter = "window's size"

min_error,best_flow,best_window_size,errors = optimize_lucas_kanade(dataset_name,smallest_window,biggest_window,step)

gt_flow = middlebury.readflo('../data/' + dataset_name + '/' + dataset["groundtruth"][dataset_name])

ang_errors = angular_error(gt_flow,best_flow["ang"])

plt.figure(figsize=(10,5))
plt.hist(ang_errors.flatten(),bins=10)
plt.title("Histogram of the errors of the best flow obtained after an optimization based on angular error")
plt.show()


epe_errors = endPoint_error(gt_flow,best_flow["EPE"])

plt.figure(figsize=(10,5))
plt.hist(epe_errors.flatten(),bins=10)
plt.title("Histogram of the errors of the best flow obtained after an optimization based on EndPoint Error")
plt.show()


norm_errors = relative_norm_error(gt_flow,best_flow["norm"])

plt.figure(figsize=(10,5))
plt.hist(norm_errors.flatten(),bins=10)
plt.title("Histogram of the errors of the best flow obtained after an optimization based on relative norm error")
plt.show()

display_results(smallest_window,biggest_window,step,best_flow,errors,involved_parameter,best_window_size,dataset_name)