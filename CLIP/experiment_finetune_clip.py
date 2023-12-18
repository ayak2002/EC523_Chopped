import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import clip
from model import ClipClassify
import json
import time
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class experiments:
    # Initializer / Instance Attributes
    def __init__(self, learning_rate, epochs, data_path_train, data_path_test, device, weight_name = "weights", EMA = 0, freeze = True, LLRD = 0.001):
        self.optimizer = None
        self.freeze = freeze
        self.learning_rate = learning_rate
        self.LLRD = LLRD
        self.epochs = epochs
        #self.data_path = data_path
        self.graph_name = weight_name 
        self.weight_name = weight_name + '.pth'
        self.device = device
        self.EMA = EMA
        self.time_elapsed = []

        self.loss = [] #intialize empty list of losses

        self.model, self.preprocess = clip.load('ViT-B/32', device)

        self.parameters = []
        self.net = ClipClassify(self.model).to(device)

        self.test_path = data_path_test
        self.path_dataset = data_path_train
        transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
        ])

        self.train_dataset = datasets.ImageFolder(self.path_dataset, transform=transform)
        self.dataloader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)

        self.test_dataset = datasets.ImageFolder(self.test_path, transform=transform)
        self.dataloader_test = DataLoader(self.test_dataset, batch_size=10, shuffle=True)
       
        self.img, self.label = next(iter(self.dataloader))

        print("initialized experiment model")


    def preprocess_images(self, inputs):
        inputs = transforms.Resize((224, 224))(inputs)
        inputs = inputs.to(self.device)
        inputs = inputs.half()
        return inputs
    
    def train(self):
        self.net.train()
        print("training")
        # Freeze Vit
        for name, param in self.net.vit.named_parameters():
            param.requires_grad = not(self.freeze)

        epochs = self.epochs
        criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        print_n = 100

        start_time = time.time()
        for epoch in range(self.epochs):
            running_loss = 0.
            for i, data in enumerate(self.dataloader):
                inputs, labels = data
                '''inputs = transforms.Resize((224, 224))(inputs)
                inputs = inputs.to(self.device)
                inputs = inputs.half()'''
                inputs = self.preprocess_images(inputs)

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
            self.loss.append(running_loss)
            end_time = time.time()
            self.time_elapsed.append(end_time - start_time)
            print(epoch," Epoch:",end_time - start_time)

    def test(self):
        model_path = 'weights/'+self.weight_name

        self.net.load_state_dict(torch.load(model_path, map_location=torch.device(self.device))['model_state_dict'])
        self.net.eval()

        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        print("Testing")
        with torch.no_grad():
            for i, data in enumerate(self.dataloader_test):
                images, labels = data
                # calculate outputs by running images through the network
                labels = labels.to(self.device)
                images = self.preprocess_images(images)
                
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
        plt.show()
    


    def save_loss(self):
        with open('losses/'+self.graph_name+'_losses.json', 'w') as file:
            json.dump(self.loss, file)

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

