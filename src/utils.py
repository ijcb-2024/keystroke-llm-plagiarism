import torch
import numpy as np
from sklearn.metrics import roc_curve
import os

def calculate_eer(d, labels):
    # Element-wise square of the difference
    with torch.no_grad() :
      # squared_diff = (fv1 - fv2) ** 2

      # # Summing the squared differences along the last dimension and taking the square root for Euclidean distance
      # d = torch.sqrt(torch.sum(squared_diff, dim=1)).detach().cpu().numpy()

      # Calculate the False Positive Rates, True Positive Rates, and thresholds
      fpr, tpr, thresholds = roc_curve(labels, d, pos_label=1)      

      # Handle cases where tpr or fpr contains nan values
      tpr = np.nan_to_num(tpr)
      fpr = np.nan_to_num(fpr)

      # Find the EER
      eer_threshold = thresholds[np.argmin(np.absolute((1 - tpr) - fpr))]
      eer = fpr[np.argmin(np.absolute((1 - tpr) - fpr))]

      return eer, eer_threshold, np.average(fpr), np.average(1-tpr)
    
def update_dict(adder, addee):
    """
    Updates a dictionary by adding the values from another dictionary.

    Parameters:
    - adder (dict): The dictionary to be updated.
    - addee (dict): The dictionary whose values will be added to the `adder` dictionary.

    Returns:
    - dict: The updated dictionary.
    """

    for key in addee:
        if key in adder:
            adder[key].extend(addee[key])
        else:
            adder[key] = addee[key]

    return adder

def split_dict(d,fraction = 0.5):
    # print(d.items())
    items = list(d.items())# Step 1: Convert to list of items
    items.sort()
    three_div = int(len(items) * fraction)
    quarter_div = int(len(items) * (1 - fraction)/2) # Step 2: Find the midpoint

    # Step 3: Split the list into two halves
    first_half_items = items[:three_div]
    second_half_items = items[three_div:three_div+quarter_div]
    third_half_items = items[three_div+quarter_div:]

    # Step 4: Convert lists back to dictionaries
    first_half_dict = dict(first_half_items)
    second_half_dict = dict(second_half_items)
    third_half_dict = dict(third_half_items)

    return first_half_dict, second_half_dict, third_half_dict

def load_latest_model(checkpoint_path) : 
  dir_list = os.listdir(os.path.join(os.getcwd(),checkpoint_path))
  num_max = 0
  
  for item in dir_list : 
    res = [int(i) for i in item if i.isdigit()]
    num_max = max(res[0],num_max)
    
  return str(num_max)

def load_model(checkpoint_path) : 
  print("Loading Model Checkpoint ....")
  curr_epoch = load_latest_model()
  model_path = "_model{}.pth".format(curr_epoch)
  print("Loading : {}".format(model_path))
  model = torch.load(os.path.join(checkpoint_path,model_path))
  return model
