import matplotlib.pyplot as plt
import middlebury
import numpy as np
from skimage import color
from skimage import io
import numpy as np

def angular_error(I_gt,I_hat):
    elem_wise_product_1 = np.multiply(I_gt[:,:,0],I_hat[:,:,0])
    elem_wise_product_2 = np.multiply(I_gt[:,:,1],I_hat[:,:,1])
    dot_product = elem_wise_product_1 + elem_wise_product_2
    
    gt_squared_norm = I_gt[:,:,0]**2 + (I_gt[:,:,1])**2
    estim_squared_norm = I_hat[:,:,0]**2 + (I_hat[:,:,1])**2
    
    tmp = (1+dot_product)/(np.sqrt(1+gt_squared_norm)*np.sqrt(1+estim_squared_norm))
    tmp[tmp > 1] = 1
    
    return np.rad2deg(np.arccos(tmp))

def endPoint_error(I_gt,I_hat):
    return np.sqrt((I_gt[:,:,0] - I_hat[:,:,0])**2 + ((I_gt[:,:,1] - I_hat[:,:,1])**2))

def relative_norm_error(I_gt,I_hat,eps = 0.00001):
    gt_norm = np.sqrt(I_gt[:,:,0]**2 + (I_gt[:,:,1])**2)
    estim_norm = np.sqrt(I_hat[:,:,0]**2 + (I_hat[:,:,1])**2)
    
    return (gt_norm - estim_norm)/(gt_norm + eps)

def quiver(flow,title,scale,step=5,eps = 0.0000000001):
    plt.figure(figsize=(10,5))
    
    plt.title(title)
    
    if(scale):
        norm = np.sqrt(flow[:,:,0]**2 + (flow[:,:,1])**2)
        flow[:,:,0] /= (norm + eps)
        flow[:,:,1] /= (norm + eps)
    
    plt.quiver(np.arange(0,flow.shape[1],step), 
               np.arange(flow.shape[0], 0,-step), 
               flow[::step,::step, 0], 
               -flow[::step,::step, 1])
    
    plt.show()

dataset = {}

dataset["groundtruth"] = {}

dataset["groundtruth"]["mysine"] = "correct_mysine.flo"
dataset["groundtruth"]["rubberwhale"] = "correct_rubberwhale10.flo"
dataset["groundtruth"]["square"] = "correct_square.flo"
dataset["groundtruth"]["yosemite"] = "correct_yos.flo"

dataset["image 1"] = {}
dataset["image 1"]["mysine"] = "mysine10.png"
dataset["image 1"]["rubberwhale"] = "frame11.png"
dataset["image 1"]["square"] = "square9.png"
dataset["image 1"]["yosemite"] = "yos10.png"

dataset["image 2"] = {}
dataset["image 2"]["mysine"] ="mysine9.png" 
dataset["image 2"]["rubberwhale"] = "frame10.png"
dataset["image 2"]["square"] = "square10.png"
dataset["image 2"]["yosemite"] = "yos9.png"

def display_results(start,stop,step,best_flow,errors,involved_parameter,best_parameters,dataset_name):
    plt.plot(list(np.arange(start,stop,step)), np.array(errors["mean"]["Angular error"]), color="dodgerblue")
    plt.fill_between(list(np.arange(start,stop,step)), np.array(errors["mean"]["Angular error"])-np.array(errors["std"]["Angular error"]), np.array(errors["mean"]["Angular error"])+np.array(errors["std"]["Angular error"]),color="lightskyblue")
    
    if(involved_parameter == "window's size"):
        plt.xlabel("Window's size",fontsize=14)
    elif(involved_parameter == "alpha"):
        plt.xlabel(r"$\alpha$",fontsize=14)
    
    plt.title(r"Means ($\pm$ std) obtained after an optimization based on angular error")
    plt.ylabel("Angular error",fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show()
    
    if(involved_parameter == "window's size"):
        quiver(best_flow["ang"], "Flow obtained after an optimization based on angular error and with a window of size " + str(best_parameters["angular error"]),scale=False)
        quiver(best_flow["ang"], "Normalized flow obtained after an optimization based on angular error and with a window of size " + str(best_parameters["angular error"]),scale=True)
    elif(involved_parameter == "alpha"):
        quiver(best_flow["ang"], r"Flow obtained after an optimization based on angular error and with $\alpha$ = " + str(round(best_parameters["angular error"],6)),scale=False)
        quiver(best_flow["ang"], r"Normalized flow obtained after an optimization based on angular error and with $\alpha$ = " + str(round(best_parameters["angular error"],6)),scale=True)
        
    
    color_map = middlebury.computeColor(best_flow['ang'])
    plt.imshow(color_map)
    
    if(involved_parameter == "window's size"):
        plt.title(r"Color map obtained after an optimization based on angular error and with a window of size " + str(best_parameters["angular error"]))
    elif(involved_parameter == "alpha"):
        plt.title(r"Color map obtained after an optimization based on angular error and with $\alpha$ = " + str(round(best_parameters["angular error"],6)))
    
    plt.show()

    plt.plot(list(np.arange(start,stop,step)), np.array(errors["mean"]["EndPoint error"]), color="dodgerblue")
    plt.fill_between(list(np.arange(start,stop,step)), np.array(errors["mean"]["EndPoint error"])-np.array(errors["std"]["EndPoint error"]), np.array(errors["mean"]["EndPoint error"])+np.array(errors["std"]["EndPoint error"]),color="lightskyblue")
    
    if(involved_parameter == "window's size"):
        plt.xlabel("Window's size",fontsize=14)
    elif(involved_parameter == "alpha"):
        plt.xlabel(r"$\alpha$",fontsize=14)
    
    plt.title(r"Means ($\pm$ std) obtained after an optimization based on EndPoint Error")
    plt.ylabel("EPE",fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show()
    
    if(involved_parameter == "window's size"):
        quiver(best_flow["EPE"], "Flow obtained after an optimization based on EndPoint Error and with a window of size " + str(best_parameters["EPE"]),scale=False)
        quiver(best_flow["EPE"], "Normalized flow obtained after an optimization based on EndPoint Error and with a window of size " + str(best_parameters["EPE"]),scale=True)
    elif(involved_parameter == "alpha"):
        quiver(best_flow["EPE"], r"Flow obtained after an optimization based on EndPoint Error and with $\alpha$ = " + str(best_parameters["EPE"]),scale=False)
        quiver(best_flow["EPE"], r"Normalized flow obtained after an optimization based on EndPoint Error and with $\alpha$ = " + str(best_parameters["EPE"]),scale=True)

    
    color_map = middlebury.computeColor(best_flow['EPE'])
    plt.imshow(color_map)
    
    if(involved_parameter == "window's size"):
        plt.title("Color map obtained after an optimization based on EndPoint Error and with a window of size " + str(best_parameters["EPE"]))
    if(involved_parameter == "alpha"):
        plt.title(r"Color map obtained after an optimization based on EndPoint Error and with $\alpha$ = " + str(round(best_parameters["EPE"],6)))
    
    plt.show()

    plt.plot(list(np.arange(start,stop,step)), np.array(errors["mean"]["Relative norm error"]), color="dodgerblue")
    plt.fill_between(list(np.arange(start,stop,step)), np.array(errors["mean"]["Relative norm error"])-np.array(errors["std"]["Relative norm error"]), np.array(errors["mean"]["Relative norm error"])+np.array(errors["std"]["Relative norm error"]),color="lightskyblue")
    
    if(involved_parameter == "window's size"):
        plt.xlabel("Window's size",fontsize=14)
        plt.title(r"Mean ($\pm$ std) obtained after an optimization based on relative norm error")
    elif(involved_parameter == "alpha"):
        plt.xlabel(r"$\alpha$",fontsize=14)
        plt.title(r"Mean ($\pm$ std) obtained after an optimization based on relative norm error")
        
    plt.ylabel("Relative norm error",fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show()

    if(involved_parameter == "window's size"):
        quiver(best_flow["norm"], "Flow obtained after an optimization based on relative norm error and with a window of size " + str(best_parameters["norm"]),scale=False)
        quiver(best_flow["norm"], "Normalized flow obtained after an optimization based on relative norm error and with a window of size " + str(best_parameters["norm"]),scale=True)
    elif(involved_parameter == "alpha"):
        quiver(best_flow["norm"], r"Flow obtained after an optimization based on relative norm error and with $\alpha$ = " + str(best_parameters["norm"]),scale=False)
        quiver(best_flow["norm"], r"Normalized flow obtained after an optimization based on relative norm error and with $\alpha$ = " + str(best_parameters["norm"]),scale=True)
    
    color_map = middlebury.computeColor(best_flow["norm"])
    
    plt.imshow(color_map)
    
    if(involved_parameter == "window's size"):
        plt.title("Color map obtained after an optimization based on relative norm error and with a window of size " + str(best_parameters["norm"]))
    if(involved_parameter == "alpha"):
        plt.title(r"Color map obtained after an optimization based on relative norm error and with $\alpha$ = " + str(round(best_parameters["norm"],6)))
                  
    plt.show()

    gt_flow = middlebury.readflo('../data/' + dataset_name + '/' + dataset["groundtruth"][dataset_name])
    quiver(gt_flow, "Groundtruth",scale=False)
    
    color_map = middlebury.computeColor(gt_flow)
    
    plt.imshow(color_map)
    plt.title("Real color map")
    plt.show()

