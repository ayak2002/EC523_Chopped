# XCeption Model

## Table Of Contents
[Overview](#Overview)  
[Methodology](#Methodology)   
[Model Accuracy](#model-accuracy)   
[Sources](#Sources) 

## Overview

Xception, a sophisticated Convolutional Neural Network (CNN) architecture developed by Google's esteemed research team, distinguishes itself through its efficiency in image classification tasks. An evolution from the Inception architecture, Xception introduces a groundbreaking strategy by employing depthwise separable convolutions. This innovative approach involves applying 2D convolutions to each input channel independently, followed by a point convolution. This distinctive separation of channelwise and spatial wise operations enhances the model's parameter efficiency and computational efficacy. The model's remarkable accuracy and efficiency in handling complex visual data have positioned Xception at the forefront of image classification technology. Given its state-of-the-art performance, we have chosen Xception as a benchmark to compare against our fine-tuned CLIP model. This strategic comparison aims to unravel insights into the adaptability of advanced neural network architectures, particularly when confronted with the challenges of sparse datasets and the nuances of classifying exotic foods.


## Model Accuracy

| 	Data Partition (%)	| 	Xception	| 
| 	------------- 	| 	------------- 	| 
| 	1	| 	24.61%	| 
| 	10	| 	62.67%	| 
| 	20	| 	69.50%	| 
| 	30	| 	73.30%	| 
| 	40	| 	74.82%	| 
| 	50	| 	76.66%	| 
| 	60	| 	74.53%	| 
| 	70	| 	66.60%	| 
| 	80	| 	61.40%	| 
| 	90	| 	64.30%	| 
| 	100	| 	61.30%	| 

## Sources




