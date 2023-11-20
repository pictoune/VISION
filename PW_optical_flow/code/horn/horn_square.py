from code.horn.horn import optimize_horn_schunck
from code.utils import display_results, dataset

dataset["image 1"]["square"] = "square9.png"
dataset["image 2"]["square"] = "square10.png"

alpha_params = {"smallest_alpha": 0.0001, "biggest_alpha": 0.04, "step": 0.0001}

dataset_name = "square"
involved_parameter = "alpha"

results = optimize_horn_schunck(dataset_name, **alpha_params, nb_iter=50)
min_error, best_flow, best_alpha, errors = results

display_results(
    *alpha_params.values(),
    best_flow=best_flow,
    errors=errors,
    involved_parameter=involved_parameter,
    best_parameters=best_alpha,
    dataset_name=dataset_name
)
