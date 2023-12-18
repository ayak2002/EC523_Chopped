import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import time
import clip
from model import ClipClassify, LinearProbe
import json
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from ptFolderDataset import PtFolderDataset

class Experiment:
    # Initializer / Instance Attributes
    def __init__(self, 
                 learning_rate, 
                 epochs, 
                 data_path_train, 
                 data_path_test, 
                 device='cuda',
                 weight_name='weights',
                 EMA = 0, 
                 freeze = True,
                 LLRD = 0.001):
        
        self.optimizer = None
        self.freeze = freeze
        self.learning_rate = learning_rate
        self.LLRD = LLRD
        self.epochs = epochs
        self.graph_name = weight_name 
        self.weight_name = weight_name + '.pth'
        self.device = device
        self.EMA = EMA

        self.loss = [] #intialize empty list of losses
        self.time_elapsed = [] #time elapsed for each epoch

        self.model, self.preprocess = clip.load('ViT-B/32', device)

        self.parameters = []
        self.net = LinearProbe() if self.freeze else ClipClassify(self.model) 
        self.net = self.net.to(device)

        self.test_path = data_path_test
        self.path_dataset = data_path_train
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float16)
        ])
        
        if self.freeze:
            self.train_dataset = PtFolderDataset(self.path_dataset)
            self.test_dataset = PtFolderDataset(self.test_path)
        else:
            self.train_dataset = datasets.ImageFolder(self.path_dataset, transform=transform)
            self.test_dataset = datasets.ImageFolder(self.test_path, transform=transform)
            
            for name, param in self.net.vit.named_parameters():
                param.requires_grad = not(self.freeze)

        self.dataloader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        self.dataloader_test = DataLoader(self.test_dataset, batch_size=10, shuffle=True)

        print("initialized experiment model")


    def train(self):
        self.net.train()
        print("training")

        epochs = self.epochs
        criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        print_n = 250

        N = len(self.train_dataset)

        img, lab = self.train_dataset[5]
        print(img.shape)

        start_time = time.time()
        for epoch in range(self.epochs):
            running_loss = 0.
            epoch_start_time = time.time()
            for i, data in enumerate(self.dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                if i % print_n == print_n - 1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_n:.3f}')
                    running_loss = 0.0
            current_time = time.time()
            print(f'Epcoh {epoch+1} completed with {current_time - epoch_start_time}')
            self.time_elapsed.append(current_time - start_time)
            self.loss.append(running_loss / N)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = 'weights/'+self.weight_name

        self.net.load_state_dict(torch.load(model_path, map_location=torch.device(self.device))['model_state_dict'])
        self.net.eval()

    def test(self):
        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        print("Testing")
        with torch.no_grad():
            for i, data in enumerate(self.dataloader_test):
                images, labels = data
                # calculate outputs by running images through the network
                labels = labels.to(self.device)
                images = images.to(self.device)

                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #print(i)

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def plot_loss(self):
        # Now plot the losses
        plt.plot(self.loss)
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_graphs/'+self.graph_name+'_loss_graph.png', dpi=300) 
        #plt.show()

    def save_loss(self):
        with open('losses/'+self.graph_name+'_losses.json', 'w') as file:
            json.dump(self.loss, file)

    def print_times(self):
        for i, time in enumerate(self.time_elapsed):
            print(f"Training time for epoch {i}: {time:.3f} seconds")

    def save_weights(self):
        print("Weights loaded")
        torch.save({
        'model_state_dict': self.net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        # Include other elements like epoch, loss, etc., if necessary
        }, ('weights/'+self.weight_name))

    # Special method for string representation (optional)
    def __str__(self):
        return f"MyClass with attribute1: {self.attribute1} and attribute2: {self.attribute2}"

