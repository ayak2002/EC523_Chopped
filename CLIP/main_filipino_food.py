import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import clip
from experiment_LLRD import Experiment
import model
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def main():
    # Your code goes here
    device = 'cuda'

    experiment_10 = Experiment(
        learning_rate = 1e-3, 
        epochs = 200, #200 gets 94.4
        data_path_train = "/projectnb/ec523kb/projects/chopped_data/filipino_food_data/train", 
        data_path_test = "/projectnb/ec523kb/projects/chopped_data/filipino_food_data/test",
        weight_name = '10_percent_LLRD_filipino_food',
        device = device,  
        freeze = False,
        LLRD = 0.99, 
        batch = 1)

    experiment_10.train()

    experiment_10.save_loss()

    experiment_10.plot_loss()

    experiment_10.save_weights()

    experiment_10.test()


    

if __name__ == "__main__":
    main()
