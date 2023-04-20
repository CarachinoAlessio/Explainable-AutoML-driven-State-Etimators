# ML techniques for State Estimation
This repo contains the code of my Master's Thesis. Specifically, it consists in exploring different techniques(Explanable AI, Physics Informed NN, ...) to perform State Estimation

## Current stage
Still pretty far from the finish line. Let's say project is at 10%

## Dataset
In order to be able to run the script, you first need to download the dataset at: https://drive.google.com/drive/folders/1Rn1Tnv0XAM1oODwcPImpoSrmGZTdzQrO?usp=sharing.
The experiments have been taken with the file named ```data_for_SE_case118_for_ML.mat```, contained in the MLP folder of the Drive repository.
The downloaded ```.mat``` file must be inside ```/a1```.

## What and How to run?
Before running, please make sure you pass the proper args. Check ```parser.py```.
- ```script.py``` is meant to provide an idea about how shap values can be computed;
- ```script_retraining_with_SHAP.py``` retrains the model by exploiting some information related to the shap values, similarly to what is described here: [Utilizing Explainable AI for improving the
Performance of Neural Networks](https://arxiv.org/pdf/2210.04686.pdf)