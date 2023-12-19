
# CLIP

## Table Of Contents
[Overview](#Overview)  
[Model Accuracy](#model-accuracy)   
[Sources](#Sources) 

## Overview

All models and files were executed in [BU's SCC](https://shib.bu.edu/idp/profile/SAML2/Redirect/SSO?execution=e1s1) using V100s. The prequisites to run these scripts can be found in [Sources](#Sources) 

### Linear Probing (experiment.py)
Linear Probing is the most cost-effective method for adapting CLIP’s image encoder for solving a classification problem. Although it is a naive implementation and typically yields worse performance than fine-tuning, it has proven to be capable of outperforming fine-tuning in out-of-distribution transfer [1]. We also can use this approach as a benchmark to compare our later discussed methods.

We adapt CLIP’s visual encoder backbone by adding a classification head that takes in as input the image vector embeddings and outputs class scores. The structure consists of a single linear layer followed by a softmax. For training, we freeze all the layers of the image encoder backbone and only update the weights of the linear layer. For all our experiments, we use a cross-entropy loss and Adam optimizer. One optimization we can use is to precompute the vector embeddings for all images in our dataset which we store as temporary files. We found that following this heuristic gave us about ten times improvement in our training time using Nvidia V100s, from a minute to about six seconds per epoch. For all partitions of our dataset, our training time until convergence with this method did not exceed ten minutes.

### Unfreezing CLIP Layers (experiment_finetune.py)
Keeping our model architecture constant, we experimented with unfreezing various layers of the visual encoder. We experimented with optimizing layers by grouping them based on the depth and type of the layer. Interestingly, when we tampered with the attention and MLP blocks we found that we quickly ran into vanishing gradient issues making it difficult to reach convergence at all. Therefore, we decided to focus our experiments primarily on fine-tuning layer normalization blocks. Although it is still unclear why layer normalization improves performance and optimization, we motivated our methods based on the commonly accepted understanding that layer normalization helps to mitigate the effects of internal covariate shift as well as improve generalization results. Therefore, fine-tuning these layer normalizations will help with the domain shift to the exotic food datasets.

### Layer-wise Learning Rate Decay (experiment_LLRD.py)
Layer-wise Learning Rate Decay (LLRD) is a common technique used in transfer learning and has been shown to improve performance when fine-tuning large language transformer models. When fine-tuning a model by unfreezing layers, there is a high risk of destroying the features learned during pre-training. On the other hand, by simply performing linear probing, we risk not allowing our model to effectively learn our new data distribution. 

## Model Accuracy

| 	Data Partition (%)	| 	Fine Tuned CLIP (Frozen)	| 	"Fine Tuned CLIP (Unfrozen LLRD layers 11-7)"	| 
| 	------------- 	| 	------------- 	| 	------------- 	| 
| 	1	| 	63%	| 	68%	| 
| 	10	| 	72%	| 	79%	| 
| 	20	| 	73%	| 	80%	| 
| 	30	| 	76%	| 	81%	| 
| 	40	| 	69%	| 	81%	| 
| 	50	| 	69%	| 	82%	| 
| 	60	| 	66%	| 	67%	| 
| 	70	| 	59%	| 	62%	| 
| 	80	| 	70%	| 	75%	| 
| 	90	| 	57%	| 	63%	| 
| 	100	| 	54%	| 	61%	| 

## Sources

First, follow this GitHub link to download the CLIP model:
https://github.com/openai/CLIP

Then, clone this repository and run any of the main.py files.
*Important:* Change the file directory paths in the main.py files to the appropiate file directories that training and testing data are located.

