# Deep Learning Dose Activity Dictionary ☄️

This repository is dedicated to the development and testing of deep learning models to predict deposited dose from PET activity measurements in proton therapy. This is part of the PROTOTWIN project, which aims to develop real-time proton range correction techniques.

## Table of Contents 📑
1. [Images](#images)
2. [Models](#models)
3. [Testing](#testing)
4. [Training](#training)
5. [Scripts and Utilities](#scripts-and-utilities)
6. [Old Files](#old-files)

## Images 🖼️
- Includes:
  - depth-dose profiles comparing the predicted and the ground truth dosages
  - slices comparing the predicted and the ground truth dosages for a given activity
  - training and validation losses vs number of epochs
  - recursive application of the direct and reverse model as a sanity check
  
## Models 🤖
- SwinUNETR.py: Implementation of the SwinUNETR model adapted from MONAI.
- TransBTS.py: Code for the TransBTS model from the [official GitHub page](https://github.com/Rubics-Xuan/TransBTS).
- models.py: u-nets and attention u-nets.

## Testing 🔍
- test-results: Folder containing test metrics for different models from model testing.
- test_model.py: Script to test trained models.

## Training 🏋️‍♂️
- training-times: Data and logs related to the training duration.
- train_model.py: Main script for model training.

## Scripts and Utilities ⚙️
- dad-nn.ipynb: Jupyter notebook going through the dataset generation, training and testing. 
- dad-nn.py: Python script version of the above notebook.
- dataset_gen.py: Script to generate the input and output image folders from the raw data (data not included because of size).
- utils.py: Various utility functions and helpers, including custom transforms, dataset classes and plotting functions, among others.

## Old Files 📦
- Archive of older versions and unused files. Not for current use.

---

Feel free to contribute and raise issues! 🌟
