from horn import optimize_horn_schunck
from utils import display_results

smallest_alpha = 0.001
biggest_alpha = 0.4
step = 0.005

dataset_name = "mysine"
involved_parameter = "alpha"

min_error,best_flow,best_alpha,errors = optimize_horn_schunck(dataset_name,smallest_alpha,biggest_alpha,step,nb_iter=200)

display_results(smallest_alpha,biggest_alpha,step,best_flow,errors,involved_parameter,best_alpha,dataset_name)