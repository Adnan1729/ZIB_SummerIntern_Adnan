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
- [Existing Tools](#Existing-Tools)

## Introduction
I am a summer intern at Zuse Institute Berlin (ZIB), aiming to learn about the domain of Explainable AI.

## Acknowledgements
I wish to express my profound gratitude to Turan Berkant for his invaluable supervision and Dr. Stephan Walchand for allowing me this internship opportunity.

## Getting Started

### Dependencies
Check the environment.yml file. 
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

## Existing Tools

---

### IoU Definitions

The core difference between the two methods is how they measure overlap: 
- **IoU_1**: Measures pixel-level overlap.
- **IoU_2**: Measures bounding box-level overlap.

### IoU_1: Pixel-Level Overlap

1. **Binary Mask from Heatmap**: Create a mask where values above a certain threshold are set to 1, and the rest to 0.
2. **Binary Mask from Bounding Boxes**: Create a mask based on the provided bounding boxes.
3. **Intersection**: Multiply the two binary masks element-wise. The sum represents the number of overlapping pixels.
4. **Union**: Add the two binary masks and count the non-zero pixels.
5. **IoU Calculation**: Compute the ratio of the intersection to the union.

### IoU_2: Bounding Box-Level Overlap

1. **Bounding Box from Heatmap**: Extract a box from the binary heatmap mask that encloses all "activated" areas.
2. **Intersection Calculation**: Determine the overlap between the heatmap's bounding box and the ground truth bounding box. Use the maximum starting x and y coordinates and the minimum ending x and y coordinates of the two bounding boxes to compute the dimensions of the intersection. Then, calculate the area as the product of its width and height.
3. **Union Calculation**: Find the combined area of the two bounding boxes and subtract the intersection area.
4. **IoU Calculation**: Compute the ratio of the intersection area to the union area.

---

### Key Parameters of Existing Explainable Tools

| Method                 | Parameter               | Description                                                                                           |
|------------------------|-------------------------|-------------------------------------------------------------------------------------------------------|
| Integrated Gradients   | `n_steps`               | Number of steps in the path integral approximation. Larger values can be more accurate but intensive. |
| Saliency               | -                       | No parameters to tune.                                                                                |
| DeepLift               | `baseline`              | A reference input to compare each input to. Choice affects the attributions.                          |
| DeepLiftShap           | `baseline`              | Same as DeepLift, the baseline reference can be important.                                            |
| GradientShap           | `baselines`             | Randomly sampled reference inputs.                                                                     |
|                        | `n_samples`             | Number of random samples per input sample.                                                             |
| Input X Gradient       | -                       | No specific parameters to optimize.                                                                    |
| Guided Backprop        | -                       | No specific parameters to optimize.                                                                    |
| Guided GradCAM         | `layer`                 | Specific layer for computing GradCAM. Affects resolution and focus of attributions.                    |
| Deconvolution          | -                       | No specific parameters to optimize.                                                                    |
| Feature Ablation       | -                       | No specific parameters to optimize.                                                                    |
| Occlusion              | `sliding_window_shapes` | Size of the patch to occlude parts of the input.                                                       |
|                        | `strides`               | Step size for the sliding window.                                                                      |
| Feature Permutation    | -                       | No specific parameters to optimize.                                                                    |
| Shapley Value Sampling | `n_samples`             | Number of Monte Carlo samples for Shapley values.                                                      |
| Lime                   | `n_samples`             | Number of perturbed samples for the surrogate model.                                                   |
|                        | `perturbations_per_eval`| Number of perturbations computed simultaneously, balancing computation vs. memory usage.               |
| KernelShap             | `n_samples`             | Number of samples to average over.                                                                     |
|                        | `baselines`             | Baseline samples for comparison.                                                                       |
| LRP                    | `rule`                  | Propagation rule, e.g., "Z" or "Z^+". Affects relevance distribution.                                 |

---

---

## Integrated Gradients

<p align="center">
  <img src="Picture5.png" width="45%" alt="Description of Picture5">
  <img src="Picture6.png" width="45%" alt="Description of Picture6">
</p>

<p align="center">
  <i><b>Figure 1</b>: Comparative visualization of original images and their corresponding heatmaps generated by Integrated Gradients. The left column showcases the original images, while the right column presents the heatmaps. The color intensity indicates the contribution magnitude, with the colorbar differentiating between positive (red) and negative (blue) contributions. Bounding boxes highlight regions of interest: the green bounding box represents the ground truth, whereas the red bounding box is derived from the heatmap.</i>
</p>

### Metrics:

- **Number of Activated Pixels**:
  - Left Set: 50153 (out of 224x224)
  - Right Set: 50125 (out of 224x224)

- **IoU_1**:
  - Left Set: 0.11006321012973785
  - Right Set: 0.1557905226945877

- **IoU_2**:
  - Left Set: 0.1110016256570816
  - Right Set: 0.1570311039686203

---






