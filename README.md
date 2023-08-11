# SVHN Dataset Classifier with Explainable AI

This repository encompasses a sophisticated neural network designed for the classification of the Street View House Numbers (SVHN) dataset. The primary objective extends beyond mere accuracy; it seeks to embed model explainability, a critical facet imperative for various professional applications.

## Table of Contents
- [Introduction](#introduction)
- [Acknowledgements](#acknowledgements)
- [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Setup](#setup)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results & Visualization](#results--visualization)
- [Weights & Biases Integration](#weights--biases-integration)

## Introduction
I am a summer intern at Zuse Institute Berlin (ZIB), aiming to learn about the domain of Explainable AI.

## Acknowledgements
I wish to express my profound gratitude to Turan Berkant for his invaluable supervision and Dr. Stephan Walchand for allowing me this internship opportunity.

## Getting Started

### Dependencies
- Python 3.x
- PyTorch
- torchvision
- pandas
- h5py
- numpy
- matplotlib
- PIL (Pillow)
- imblearn
- scikit-learn
- wandb (Weights & Biases)

### Setup
N/A (will update later)

## Model Architecture
The underlying architecture is predicated on the ResNet-18 model. It has been tailored to facilitate binary classification, with a primary focus on identifying a specific target digit within the customised SVHN dataset.

## Usage
To employ the model, at this stage just execute the primary Python script (main.py).


## Results & Visualization
Upon the culmination of the training process, a graphical representation will be generated, elucidating the training and test losses, in addition to accuracies, plotted against the epochs. 

## Weights & Biases Integration
The framework has been integrated with Weights & Biases for meticulous experiment tracking. It is imperative to have an active Weights & Biases account and a designated project to effectively log and monitor the experiments.



