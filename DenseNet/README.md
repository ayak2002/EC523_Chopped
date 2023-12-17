# DenseNet

## Table Of Contents
[Overview](#Overview)  
[Methodology](#Methodology)  
[Model Accuracy](#model-accuracy)   
[Sources](#Sources) 

## Overview

DenseNet-201, a robust convolutional neural network (CNN) architecture, stands as another contender in our comparative analysis for its performance in image classification tasks. Developed as an extension of the DenseNet family, this architecture is distinguished by its dense connectivity pattern, where each layer receives direct input from all preceding layers. This unique structure fosters feature reuse and enhances model accuracy. DenseNet-201, with its 201 layers, delves deeper into the extraction of intricate features, making it particularly adept at discerning complex patterns within diverse datasets. Moreover, an additional advantage of DenseNet-201 compared to other DenseNet models lies in its relatively lighter computational load, making it more resource-efficient. This characteristic is particularly valuable in scenarios where computational resources may be limited. Its comprehensive feature extraction capabilities, coupled with the efficient use of parameters, position DenseNet-201 as a suitable candidate for evaluating the performance of our fine-tuned CLIP model. 

## Methodology

### Pre-Trained Model

We used DenseNet that was pre-trained on ImageNet, a large visual database designed for use in visual object recognition research. This pre-training allows DenseNet to learn a wide variety of features and patterns present in a diverse set of images.

### No Weights

To create a baseline of DenseNet, we trained it with no weights. By doing this, we received extremely low accuracies in comparison to those with the ImageNet Pre-trained weights. 

## Model Accuracy

| 	Data Partition (%)	| 	Dense Net, weights = imagenet(15 epochs)	| 	DenseNet, weights = None	| 
| 	------------- 	| 	------------- 	| 	------------- 	| 
| 	1	| 	32.10%	| 	2.02%	| 
| 	10	| 	56%	| 	5.89%	| 
| 	20	| 	59.94%	| 	9.06%	| 
| 	30	| 	62%	| 	9.67%	| 
| 	40	| 	63%	| 	10.01%	| 
| 	50	| 	64%	| 	10.13%	| 
| 	60	| 	61%	| 	11.67%	| 
| 	70	| 	56%	| 	10.74%	| 
| 	80	| 	53%	| 	9.59%	| 
| 	90	| 	53.54%	| 	9.48%	| 
| 	100	| 	52%	| 	8.63%	| 

## Sources
[Food-101 High Accuracy](https://www.kaggle.com/code/khadijacheema/food-101-high-accuracy96-67) 



