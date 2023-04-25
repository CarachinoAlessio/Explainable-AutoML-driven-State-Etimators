# ML techniques for State Estimation
This repo contains the code of my Master's Thesis. Specifically, it consists in exploring different techniques(Explanable AI, Physics Informed NN, ...) to perform State Estimation

## Current stage
Still pretty far from the finish line. Let's say project is at 10%

## Dataset
In order to be able to run the script, you first need to download the dataset at: https://drive.google.com/drive/folders/1Rn1Tnv0XAM1oODwcPImpoSrmGZTdzQrO?usp=sharing.
The experiments have been taken with the file named ```data_for_SE_case118_for_ML.mat```, contained in the MLP folder of the Drive repository.
The downloaded ```.mat``` file must be inside ```/case118```.

## What and How to run?
- ```script.py``` is meant to provide an idea about how shap values can be computed and what kind of explanations can be generated;
- ```script_retraining_with_SHAP.py``` retrains the model by exploiting some information related to the shap values, similarly to what is described here: [Utilizing Explainable AI for improving the
Performance of Neural Networks](https://arxiv.org/pdf/2210.04686.pdf)

### How to run ```script.py```
- First, you need to general the model (```.pth```):
  - ```python script.py --train True```
- Then, you can load the trained model and use it to generate the shap values
  - ```python script.py --shap_values True```
- Finally, you can load the trained model and the shap values to generate explanations
  - ```python script.py```

### How to run ```script_retraining_with_SHAP.py```
- First, you need to general the model (```.pth```):
  - ```python script_retraining_with_SHAP.py --train True```
- Then, you need to apply the retraining procedure (described above)
  - ```python script_retraining_with_SHAP.py --retrain_time True```
- Finally, you can load the retrained model and perform again the tests
  - ```python script_retraining_with_SHAP.py --test_retrained True```