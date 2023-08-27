from horn import optimize_horn_schunck
from utils import display_results, dataset

dataset["image 1"]["square"] = "square9.png"
dataset["image 2"]["square"] = "square10.png"

smallest_alpha = 0.0001
biggest_alpha = 0.04
step = 0.0001

dataset_name = "square"
involved_parameter = "alpha"

min_error,best_flow,best_alpha,errors = optimize_horn_schunck(dataset_name,smallest_alpha,biggest_alpha,step,nb_iter=50)

display_results(smallest_alpha,biggest_alpha,step,best_flow,errors,involved_parameter,best_alpha,dataset_name)