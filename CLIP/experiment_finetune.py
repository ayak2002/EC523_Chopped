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

class Experiment:
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

        base_lr = 0.001
        decay_factor = 0.9  # Less than 1.0 for decay

        # Create a list to hold parameter groups
        self.param_groups = []



        # Iterate through the model's layers and apply decay


        self.parameters = []
        self.net = ClipClassify(self.model).to(device)

        self.optimizers = {}

        self.layer_lr = self.learning_rate

        
        learning_rates = {
            'base': 0.001,  # Default learning rate for layers not explicitly mentioned
            'blocks': {
            'vit.transformer.resblocks.0': 0.0001,
            'vit.transformer.resblocks.1': 0.0001,
            # ... Add learning rates for other blocks as needed
            }
        }

        self.param_groups = []
        base_params = {'params': []}
        for name, param in self.net.named_parameters():
            added = False
            for block_name, lr in learning_rates['blocks'].items():
                #print("Block name:", block_name)
                if block_name in name:
                    self.param_groups.append({'params': [param], 'lr': lr})
                    added = True
                    break
            if not added:
                base_params['params'].append(param)

        if base_params['params']:
            base_params['lr'] = learning_rates['base']
            self.param_groups.append(base_params)

        '''for name, param in self.net.named_parameters():
            print("Name:", name)
            self.optimizers[name] = torch.optim.Adam([param], lr=self.layer_lr)
            layer_l *= self.LLRD'''


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

    def decay_lr(optimizer, decay_factor):
        for self.param_group in optimizer.param_groups:
            self.param_group['lr'] *= decay_factor



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
            print("name: ", name)
            param.requires_grad = not(self.freeze)

        # First, identify the name of the last layer
        layers_to_unfreeze = [
            'transformer.resblocks.11.ln_2.weight',
            'transformer.resblocks.11.ln_2.bias',
            'transformer.resblocks.11.ln_1.weight',
            'transformer.resblocks.11.ln_1.bias',
            'transformer.resblocks.10.ln_2.weight',
            'transformer.resblocks.10.ln_2.bias',
            'transformer.resblocks.10.ln_1.weight',
            'transformer.resblocks.10.ln_1.bias',
            'transformer.resblocks.9.ln_2.weight',
            'transformer.resblocks.9.ln_2.bias',
            'transformer.resblocks.9.ln_1.weight',
            'transformer.resblocks.9.ln_1.bias',
            'transformer.resblocks.8.ln_2.weight',
            'transformer.resblocks.8.ln_2.bias',
            'transformer.resblocks.8.ln_1.weight',
            'transformer.resblocks.8.ln_1.bias',
            'transformer.resblocks.7.ln_2.weight',
            'transformer.resblocks.7.ln_2.bias',
            'transformer.resblocks.7.ln_1.weight',
            'transformer.resblocks.7.ln_1.bias',
            'ln_post.weight'
            'ln_post.bias'
        ]

        # Iterate through the parameters and unfreeze the specified layers
        '''for name, param in self.net.vit.named_parameters():
            if name in layers_to_unfreeze:
                param.requires_grad = True
                print(f"Unfroze layer: {name}")
            else:
                param.requires_grad = False
                print(f"Froze layer: {name}")'''

            
                

        # Optionally, check if the last layer is unfrozen
        '''for name, param in self.net.vit.named_parameters():
            print(f"Layer: {name}, Frozen: {not param.requires_grad}")'''
        


        epochs = self.epochs
        criterion = nn.CrossEntropyLoss()
        print("Net param: ", self.net.parameters())
        #self.optimizer = torch.optim.Adam(self.param_groups)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()))#self.net.parameters(), lr=self.learning_rate)
        #print(self.param_groups)
        print_n = 100
        for param_g in self.optimizer.param_groups:
            print(param_g['lr'])

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

                '''for name, parameter in self.net.named_parameters():
                    if parameter.grad is not None:
                        print(f"{name} - Gradient: {parameter.grad}")
                else:
                    print(f"{name} - No gradient")'''

                self.optimizer.step()

                #break  

                # print statistics
                running_loss += loss.item()
                #print(loss.item())
                
                if i % print_n == print_n - 1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_n:.3f}')
                    running_loss = 0.0
            #if(epoch == 1):
                #break
            self.loss.append(running_loss)
            end_time = time.time()
            self.time_elapsed.append(end_time - start_time)
            print((epoch+1)," Epoch:",end_time - start_time)

    def test(self):
        model_path = 'weights/'+self.weight_name

        self.net.load_state_dict(torch.load(model_path, map_location=torch.device(self.device))['model_state_dict'])
        self.net.eval()

        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        print("Testing")
        start_time = time.time()
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

        end_time = time.time()
        print(" Time:",end_time - start_time)
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
