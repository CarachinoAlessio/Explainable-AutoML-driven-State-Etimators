# ML techniques for State Estimation
This repo contains the code of my Master's Thesis. Specifically, it aims to validating the usefulness of some AutoML frameworks in the Smart Grid context and providing an overview of Explainable AI techniques employed in such context. 

---

# System Specifications

This section outlines the system requirements and specifications needed to run the application.


### Hardware specifications
- **Processor**: 13th Gen Intel(R) Core(TM) i5-13500
- **GPU**: NVIDIA GeForce RTX 4070 with 12 GB VRAM
- **Memory**: 32GB 6600Mhz DDR5.
- **Python Version**: Python 3.10.
- **Dependencies**: Other dependencies are declared in the ```requirements.txt``` file.

To be fair, it should be noted that some tests were conducted on a WSL machine (running Ubuntu). Specifically, Autosklearn and Autogluon were tested on WSL, while MLP and H2O were tested on Windows.
Every other sub-task, from data generation through pandapower to voltage magnitude estimation plots were performed on Windows.

The exact same results described in the Thesis are partially reproducible since many of are base on a ```time_limit``` parameter. However, results without the time_limit parameter should be perfectly reproducible if the libraries versions match the ones I defined in the requirements files. 

---

## Dataset
The tests were conducted on two distinct grids, referred to as net18 and net95. For each grid, four variants were tested: net18v1, net18v2, net18v3, net18v4, and similarly, net95v1, net95v2, net95v3, net95v4. The data for each variant was obtained through simulations conducted with pandapower, a Python tool for power system analysis.

Additionally, some extra validation data files were used to further ensure the robustness and accuracy of the results.

All net18 variants are characterized by having 18 output nodes, while all net95 variants have 95 output nodes. The number of input nodes differs across each variant, tailored to specific configurations and requirements. The tests involve performing a state estimation task, which assesses the estimators ability to accurately estimate the system's state based on the available measurements.

Each variant is characterized by the following number of measurements (input for the estimator):

- **net18v1**: 53.
- **net18v2**: 40.
- **net18v3**: 43.
- **net18v4**: 53.

- **net95v1**: 206.
- **net95v2**: 206.
- **net95v3**: 206.
- **net95v4**: 206.


---



## What and How to run?
First of all, dependencies have to be installed by running 
```bash
pip install -r requirements[Windows|WSL].txt
```
Please, make sure to pick between Windows and WSL according to the kind of test you want to perform.

In the directory ```'./examples/'``` you can find some READY-TO-USE Jupyter Notebook to perform tests on net95v1. Change the parameters accordingly to your needs. 