
# CLIP

## Table Of Contents
[Overview](#Overview)  
[Model Accuracy](#model-accuracy)   
[Sources](#Sources) 

## Overview

All models and files were executed in BU's SCC using V100s. The prequisites to run scripts are in the [Sources](#Sources) 

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

