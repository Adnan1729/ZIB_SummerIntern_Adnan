import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid
from typing import Optional, Callable
from captum.attr import LRP, KernelShap, Lime, ShapleyValueSampling, FeaturePermutation, Occlusion, FeatureAblation, Deconvolution, GuidedGradCam, GuidedBackprop, InputXGradient, GradientShap, DeepLiftShap, DeepLift, Saliency, IntegratedGradients
from dataset_custom_01 import SVHNDataset
# =========================
# Constants
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = "/mnt/c/Users/adnan/OneDrive/Documents/SVHNDataset"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
TARGET_DIGIT = 1
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
TARGET_TRANSFORM = None # Torch backend handels it
BALANCE = True
MAX_SAMPLES = 20
ONE_VS_TWO = None
USE_GRAYSCALE = None
BATCH_SIZE = 1

# =========================
# Custom Dataset
# =========================

# Being imported from dataset_custom.py
    
# =========================
# Model Loading
# =========================

def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

# =========================
# Attribution Function
# =========================

def get_lrp_attributions(model, data_loader):
    lrp = LRP(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = lrp.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_kernelshap_attributions(model, data_loader):
    kernel_shap = KernelShap(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    baselines = torch.zeros(*inputs.shape).to(device)
    
    attributions = kernel_shap.attribute(inputs, baselines, target=labels, n_samples=200)
    
    return inputs, attributions, bounding_boxes


def get_lime_attributions(model, data_loader):
    lime = Lime(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = lime.attribute(inputs, target=labels, n_samples=125, perturbations_per_eval=10)
    
    return inputs, attributions, bounding_boxes


def get_shapley_value_attributions(model, data_loader):
    shapley = ShapleyValueSampling(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = shapley.attribute(inputs, target=labels, n_samples=25, perturbations_per_eval=5)
    
    return inputs, attributions, bounding_boxes


def get_feature_permutation_attributions(model, data_loader):
    feature_permutation = FeaturePermutation(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = feature_permutation.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_occlusion_attributions(model, data_loader):
    occlusion = Occlusion(model)
    
    # The size of the sliding window used to occlude parts of the input. Needs to be adjust this based on specific use case.
    sliding_window_shapes = (3, 15, 15)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = occlusion.attribute(inputs, strides = (3, 8, 8),
                                       target=labels, sliding_window_shapes=sliding_window_shapes)
    
    return inputs, attributions, bounding_boxes


def get_feature_ablation_attributions(model, data_loader):
    feature_ablation = FeatureAblation(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = feature_ablation.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_deconvolution_attributions(model, data_loader):
    deconv = Deconvolution(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = deconv.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_guidedgradcam_attributions(model, data_loader):
    # Specify the layer for GradCAM 
    target_layer = model.layer4[-1]  # for ResNet18, the final convolutional layer is layer4[-1]
    
    guided_gc = GuidedGradCam(model, target_layer)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = guided_gc.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_guidedbackprop_attributions(model, data_loader):
    guided_backprop = GuidedBackprop(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = guided_backprop.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_inputxgradient_attributions(model, data_loader):
    input_x_gradient = InputXGradient(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = input_x_gradient.attribute(inputs, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_gradientshap_attributions(model, data_loader):
    gradient_shap = GradientShap(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Choosing baselines randomly
    baselines = torch.randn(20, 3, 128, 128).to(device)  # Adjusted the dimensions to 128x128 as per your transformation
    attributions = gradient_shap.attribute(inputs, baselines=baselines, target=labels)
    
    return inputs, attributions, bounding_boxes


def get_deepliftshap_attributions(model, data_loader):
    dl_shap = DeepLiftShap(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    baselines = torch.zeros((10,) + inputs.shape[1:]).to(device)  # 10 baselines for DeepLiftShap
    attributions = dl_shap.attribute(inputs, baselines=baselines, target=labels)
    return inputs, attributions, bounding_boxes


def get_deeplift_attributions(model, data_loader):
    dl = DeepLift(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = dl.attribute(inputs, target=labels)
    return inputs, attributions, bounding_boxes


def get_saliency_attributions(model, data_loader):
    saliency = Saliency(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = saliency.attribute(inputs, target=labels)
    return inputs, attributions, bounding_boxes


def get_IntegratedGradients_attributions(model, data_loader):
    integrated_gradients = IntegratedGradients(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = integrated_gradients.attribute(inputs, target=labels, n_steps=100)
    return inputs, attributions, bounding_boxes


# =========================
# IoU Calc
# =========================

def threshold_heatmap(heatmap: torch.Tensor, k: int) -> torch.Tensor:
    """
    Threshold the heatmap to retain only the top k pixels.
    
    Args:
    - heatmap (torch.Tensor): The heatmap tensor.
    - k (int): Number of top pixels to retain.
    
    Returns:
    - torch.Tensor: A binary heatmap with top k pixels set to 1.
    """
    # Flatten the heatmap and get the value of the k-th largest pixel
    threshold_value = heatmap.flatten().topk(k)[0][-1]
    return (heatmap >= threshold_value).float()

def calculate_iou(heatmap: torch.Tensor, bbox: torch.Tensor, k: int) -> float:
    """
    Calculate the Intersection over Union (IoU) between a thresholded heatmap and a bounding box.
    
    Args:
    - heatmap (torch.Tensor): The heatmap tensor.
    - bbox (torch.Tensor): Binary tensor representing the bounding box. Should be the same shape as heatmap.
    - k (int): Number of top pixels in the heatmap to consider.
    
    Returns:
    - float: The IoU value.
    """
    # Threshold the heatmap
    thresh_heatmap = threshold_heatmap(heatmap, k)
    
    # Calculate intersection and union
    intersection = (thresh_heatmap * bbox).sum()
    union = thresh_heatmap.sum() + bbox.sum() - intersection
    
    # Return IoU
    return (intersection / union).item()


# =========================
# Processing Attribution Function
# =========================

def save_attributions(inputs, attributions, save_dir, method_name):
    for i, input in enumerate(inputs):
        original_img = (input.cpu() - input.cpu().min()) / (input.cpu().max() - input.cpu().min())
        method_map = (attributions[i].cpu() - attributions[i].cpu().min()) / (attributions[i].cpu().max() - attributions[i].cpu().min())
        concatenated_images = make_grid([original_img, method_map], nrow=2)
        save_image(concatenated_images, os.path.join(save_dir, f"{method_name.lower()}_{i}.png"))



# =========================
# Parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Explainability Methods for Custom SVHN dataset")
    parser.add_argument("--method", type=str, required=True, choices=["LRP", "KernelShap", "Lime", "ShapleyValueSampling", "FeaturePermutation", "Occlusion", "FeatureAblation", "Deconvolution", "GuidedGradCam", "GuidedBackprop", "InputXGradient", "GradientShap", "DeepLiftShap", "DeepLift", "Saliency", "IntegratedGradients"], help="Which explainability method to run.")
    return parser.parse_args()

# =========================
# Execution 
# =========================

if __name__ == "__main__":
    save_file_path = "/mnt/c/Users/adnan/OneDrive/Documents/SVHNDataset/attributions"
    model_path = "/mnt/c/Users/adnan/OneDrive/Documents/SVHNDataset/model_weights.pth"
    args = parse_args()
    print(f"Method selected: {args.method}")
    model = load_model(model_path)
    dataset = SVHNDataset(FILE_PATH, TEST_SPLIT, TARGET_DIGIT, TRANSFORM, TARGET_TRANSFORM, BALANCE, MAX_SAMPLES, ONE_VS_TWO, USE_GRAYSCALE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    method_to_function = {
        "LRP": get_lrp_attributions,
        "KernelShap": get_kernelshap_attributions,
        "Lime": get_lime_attributions,
        "ShapleyValueSampling": get_shapley_value_attributions,
        "FeaturePermutation": get_feature_permutation_attributions,
        "Occlusion": get_occlusion_attributions,
        "FeatureAblation": get_feature_ablation_attributions,
        "Deconvolution": get_deconvolution_attributions,
        "GuidedGradCam": get_guidedgradcam_attributions,
        "GuidedBackprop": get_guidedbackprop_attributions,
        "InputXGradient": get_inputxgradient_attributions,
        "GradientShap": get_gradientshap_attributions,
        "DeepLiftShap": get_deepliftshap_attributions,
        "DeepLift": get_deeplift_attributions,
        "Saliency": get_saliency_attributions,
        "IntegratedGradients": get_IntegratedGradients_attributions
    }

    inputs, attributions, bounding_boxes = method_to_function[args.method](model, data_loader)  
    
    # Calculate IoU for each image's attributions and bounding box
    k = 1000  # or any value you desire
    ious = [calculate_iou(attribution.squeeze(0), bbox.squeeze(0), k) for attribution, bbox in zip(attributions, bounding_boxes)]
    
    # Optionally, print out the IoUs or store them for later analysis
    for idx, iou_val in enumerate(ious):
        print(f"Image {idx + 1}: IoU = {iou_val:.4f}")
    
    save_attributions(inputs, attributions, save_file_path, args.method)
    
