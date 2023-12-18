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

    print("learning decay 0.9")
    for b in range (2,10,2):
        print("batch size:",b)
        experiment_10 = Experiment(
        learning_rate = 1e-3, 
        epochs = 10, 
        data_path_train = "/projectnb/ec523kb/projects/chopped_data/food101_10percent/train", 
        data_path_test = "/projectnb/ec523kb/projects/chopped_data/food101_test",
        weight_name = '10_percent_LLRD_025',
        device = device,  
        freeze = False,
        LLRD = 0.99,
        batch = b
        )

        experiment_10.train()

        experiment_10.save_loss()

        experiment_10.plot_loss()

        experiment_10.save_weights()

        experiment_10.test()


if __name__ == "__main__":
    main()
