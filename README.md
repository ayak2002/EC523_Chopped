# EC523_Chopped

## Table Of Contents
[Overview](#Overview)  
[Data](#Data)   
[Model Accuracy](#model-accuracy)
[DenseNet](#DenseNet)   
[XCeption](#XCeption)   
[CLIP](#CLIP) 

## Overview

Many annotated food image datasets are biased to westernized foods, such as the food in the ImageNet. Data sets for more exotic foods such as asian fruits are sparse, making it more difficult to train conventional image neural network classifiers on these sparse datasets and having good classification accuracy. Our task is to compare performance of conventional image neural networks classifiers and fine-tuned CLIP model on increasing dataset sparsity and demonstrate that CLIP is a capable pretrained model to increase accuracy on sparse datasets. CLIP will be used for transfer learning to increase accuracy on sparse exotic food datasets. Our project focuses on food image classification and removing the NLP recipe generation section as there are many well studied and implemented models for this task. We have decided to center this project around food image classification because during our search for food datasets and found many westernized datasets and there were very few datasets on asian food, and if they did exist, the datasets were small. 

## Data
[Data Sub-Folder](https://github.com/ayak2002/EC523_Chopped/tree/main/data)
Unlike other datasets, Food-101 allows machine learning models to develop a broader understanding of global cuisine, enabling them to recognize and classify not only familiar dishes but also exotic and culturally diverse foods accurately. 

## Model Accuracy 

| 	Data Partition (%)	| Dense Net, weights = imagenet(15 epochs)	| DenseNet, weights = None	| Xception	| Fine Tuned CLIP (Frozen)	| "Fine Tuned CLIP (Unfrozen LLRD layers 11-7)"	| 
| 	------------- 	| 	------------- 	| 	------------- 	| 	------------- 	| 	------------- 	| 	------------- 	| 
| 	1	| 	32.10%	| 	2.02%	| 	24.61%	| 	63%	| 	68%	| 
| 	10	| 	56%	| 	5.89%	| 	62.67%	| 	72%	| 	79%	| 
| 	20	| 	59.94%	| 	9.06%	| 	69.50%	| 	73%	| 	80%	| 
| 	30	| 	62%	| 	9.67%	| 	73.30%	| 	76%	| 	81%	| 
| 	40	| 	63%	| 	10.01%	| 	74.82%	| 	69%	| 	81%	| 
| 	50	| 	64%	| 	10.13%	| 	76.66%	| 	69%	| 	82%	| 
| 	60	| 	61%	| 	11.67%	| 	74.53%	| 	66%	| 	67%	| 
| 	70	| 	56%	| 	10.74%	| 	66.60%	| 	59%	| 	62%	| 
| 	80	| 	53%	| 	9.59%	| 	61.40%	| 	70%	| 	75%	| 
| 	90	| 	53.54%	| 	9.48%	| 	64.30%	| 	57%	| 	63%	| 
| 	100	| 	52%	| 	8.63%	| 	61.30%	| 	54%	| 	61%	| 

## DenseNet
[DenseNet Sub-Folder](https://github.com/ayak2002/EC523_Chopped/tree/main/DenseNet)

## XCeption
[XCeption Sub-Folder](https://github.com/ayak2002/EC523_Chopped/tree/main/XceptionModel)


## CLIP
[CLIP Sub-Folder](https://github.com/ayak2002/EC523_Chopped/tree/main/CLIP)

