import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import math
import clip
from model import ClipClassify
import json
import time
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class Experiment:
    # Initializer / Instance Attributes
    def __init__(self, learning_rate, epochs, data_path_train, data_path_test, device, weight_name = "weights", EMA = 0, freeze = True, LLRD = 0.001, exclude_attn=True, batch = 10):
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
        self.exclude_attn = exclude_attn

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


        self.test_path = data_path_test
        self.path_dataset = data_path_train
        transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
        ])

        self.train_dataset = datasets.ImageFolder(self.path_dataset, transform=transform)
        self.dataloader = DataLoader(self.train_dataset, batch_size=batch, shuffle=True)

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

    def get_param_groups_with_llrd(self):
        LAYERS = 12

        for param in self.net.parameters():
            param.requires_grad = False
        
        post_layer_norm = [layer for name, layer in self.net.vit.ln_post.named_parameters()]
        classification_head = [layer for name, layer in self.net.classification_head.named_parameters()]
        params = []
        params.append({'params': [*classification_head, *post_layer_norm], 'lr': self.learning_rate})
        
        for i in range(LAYERS):
            named_params = self.net.vit.transformer.resblocks[i].named_parameters()
            resblock_params = [layer for name, layer in named_params if not self.exclude_attn or 'attn' not in name]
            
            lr = self.learning_rate * math.pow(self.LLRD, LAYERS - i)
            params.append({'params': resblock_params, 'lr': lr})

        for param in params:
            for layer in param['params']:
                layer.requires_grad = True

        return params
    
    def train(self):
        self.net.train()
        print("training")
        # Freeze Vit
        for name, param in self.net.vit.named_parameters():
            #print("name: ", name)
            param.requires_grad = not(self.freeze)

        # First, identify the name of the last layer
        layers_to_unfreeze = [
            #'default',
            'classification_head.weight',
            'classification_head.bias',
            'vit.ln_post.weight',
            'vit.ln_post.bias',
            'vit.transformer.resblocks.11.ln_2.weight',
            'vit.transformer.resblocks.11.ln_2.bias',
            'vit.transformer.resblocks.11.ln_1.weight',
            'vit.transformer.resblocks.11.ln_1.bias',
            'vit.transformer.resblocks.10.ln_2.weight',
            'vit.transformer.resblocks.10.ln_2.bias',
            'vit.transformer.resblocks.10.ln_1.weight',
            'vit.transformer.resblocks.10.ln_1.bias',
            'vit.transformer.resblocks.9.ln_2.weight',
            'vit.transformer.resblocks.9.ln_2.bias',
            'vit.transformer.resblocks.9.ln_1.weight',
            'vit.transformer.resblocks.9.ln_1.bias',
            'vit.transformer.resblocks.8.ln_2.weight',
            'vit.transformer.resblocks.8.ln_2.bias',
            'vit.transformer.resblocks.8.ln_1.weight',
            'vit.transformer.resblocks.8.ln_1.bias',
            'vit.transformer.resblocks.7.ln_2.weight',
            'vit.transformer.resblocks.7.ln_2.bias',
            'vit.transformer.resblocks.7.ln_1.weight',
            'vit.transformer.resblocks.7.ln_1.bias',
            'vit.transformer.resblocks.6.ln_2.weight',
            'vit.transformer.resblocks.6.ln_2.bias',
            'vit.transformer.resblocks.6.ln_1.weight',
            'vit.transformer.resblocks.6.ln_1.bias',
            '''vit.transformer.resblocks.5.ln_2.weight',
            'vit.transformer.resblocks.5.ln_2.bias',
            'vit.transformer.resblocks.5.ln_1.weight',
            'vit.transformer.resblocks.5.ln_1.bias',
            'vit.transformer.resblocks.4.ln_2.weight',
            'vit.transformer.resblocks.4.ln_2.bias',
            'vit.transformer.resblocks.4.ln_1.weight',
            'vit.transformer.resblocks.4.ln_1.bias',
            'vit.transformer.resblocks.3.ln_2.weight',
            'vit.transformer.resblocks.3.ln_2.bias',
            'vit.transformer.resblocks.3.ln_1.weight',
            'vit.transformer.resblocks.3.ln_1.bias',
            'vit.transformer.resblocks.2.ln_2.weight',
            'vit.transformer.resblocks.2.ln_2.bias',
            'vit.transformer.resblocks.2.ln_1.weight',
            'vit.transformer.resblocks.2.ln_1.bias',
            'vit.transformer.resblocks.1.ln_2.weight',
            'vit.transformer.resblocks.1.ln_2.bias',
            'vit.transformer.resblocks.1.ln_1.weight',
            'vit.transformer.resblocks.1.ln_1.bias'''
        ]

        lr_dict = {}
        current_lr = self.learning_rate
        for layer in layers_to_unfreeze:
            lr_dict[layer] = current_lr
            current_lr *= self.LLRD


        param_groups = []
        for name, param in self.net.named_parameters():
            if name in layers_to_unfreeze:
                param.requires_grad = True
                param_groups.append({'params': param, 'lr': lr_dict[name]})
                # print(name, ": ", lr_dict[name])
            else:
                param.requires_grad = False

        '''for name, param in self.net.named_parameters():
            if name in layers_to_unfreeze:
                param_groups.append({'params': param, 'lr': 1e-3})'''

        # Define optimizer with parameter group
        '''for param_g in param_groups:
            print(param_g)'''
        if not self.exclude_attn:
            param_groups = self.get_param_groups_with_llrd()
        self.optimizer = torch.optim.Adam(param_groups)
                     

        # Optionally, check if the last layer is unfrozen
        '''for name, param in self.net.named_parameters():
            print(f"Layer: {name}, Frozen: {not param.requires_grad}")'''
        '''for param_g in self.optimizer.param_groups:
            print(param_g['lr'])'''
        


        epochs = self.epochs
        criterion = nn.CrossEntropyLoss()
        #print("Net param: ", self.net.parameters())
        #self.optimizer = torch.optim.Adam(self.param_groups)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #print(self.param_groups)
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
            print(f'[{epoch + 1}] loss: {running_loss / print_n:.3f}')
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

        with open(self.graph_name+'_accuracy.json', 'w') as file:
            json.dump((100 * correct // total), file)


        print(f' Accuracy of the network on the 10000 test images: {100 * correct / total} %')

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
