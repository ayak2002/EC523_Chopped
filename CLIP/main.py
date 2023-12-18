import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import clip
from experiment import Experiment
import model
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def main():
    # Your code goes here
    device = 'cuda'

    experiment_10 = Experiment(
        learning_rate = 1e-3, 
        epochs = 10, 
        data_path_train = "/projectnb/ec523kb/projects/chopped_data/filipino_food_data/train", 
        data_path_test = "/projectnb/ec523kb/projects/chopped_data/filipino_food_data/test",
        weight_name = 'filipino_loss',
        device = device,  
        freeze = True,
        LLRD = 0.9)

    experiment_10.train()

    experiment_10.save_loss()

    experiment_10.plot_loss()

    experiment_10.save_weights()

    experiment_10.test()
    

if __name__ == "__main__":
    main()
