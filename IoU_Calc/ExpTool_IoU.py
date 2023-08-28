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
from dataset_IoU import SVHNDataset

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

# Being imported from dataset_custom_01.py
    
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

# Parametrised

def get_lrp_attributions(model, data_loader, rule="Z"):
    lrp = LRP(model, rule=rule)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = lrp.attribute(inputs, target=labels)
    return inputs, attributions, bounding_boxes

def get_kernelshap_attributions(model, data_loader, n_samples=200, baselines=None):
    kernel_shap = KernelShap(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    if baselines is None:
        baselines = torch.zeros(*inputs.shape).to(device)
    attributions = kernel_shap.attribute(inputs, baselines, target=labels, n_samples=n_samples)
    return inputs, attributions, bounding_boxes

def get_lime_attributions(model, data_loader, n_samples=125, perturbations_per_eval=10):
    lime = Lime(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = lime.attribute(inputs, target=labels, n_samples=n_samples, perturbations_per_eval=perturbations_per_eval)
    return inputs, attributions, bounding_boxes

def get_shapley_value_attributions(model, data_loader, n_samples=25):
    shapley = ShapleyValueSampling(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = shapley.attribute(inputs, target=labels, n_samples=n_samples)
    return inputs, attributions, bounding_boxes

def get_occlusion_attributions(model, data_loader, sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8)):
    occlusion = Occlusion(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = occlusion.attribute(inputs, strides=strides, target=labels, sliding_window_shapes=sliding_window_shapes)
    return inputs, attributions, bounding_boxes

def get_gradientshap_attributions(model, data_loader, baselines=None, n_samples=5):
    gradient_shap = GradientShap(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    if baselines is None:
        baselines = torch.randn(n_samples, 3, 128, 128).to(device)
    attributions = gradient_shap.attribute(inputs, baselines=baselines, target=labels)
    return inputs, attributions, bounding_boxes

def get_deepliftshap_attributions(model, data_loader, baseline=None):
    dl_shap = DeepLiftShap(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    attributions = dl_shap.attribute(inputs, baselines=baseline, target=labels)
    return inputs, attributions, bounding_boxes


def get_deeplift_attributions(model, data_loader, baseline=None):
    dl = DeepLift(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    attributions = dl.attribute(inputs, baselines=baseline, target=labels)
    return inputs, attributions, bounding_boxes

def get_IntegratedGradients_attributions(model, data_loader, n_steps=200):
    integrated_gradients = IntegratedGradients(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = integrated_gradients.attribute(inputs, target=labels, n_steps=n_steps)
    return inputs, attributions, bounding_boxes

# Not Parametrised

def get_feature_permutation_attributions(model, data_loader):
    feature_permutation = FeaturePermutation(model)
    
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    attributions = feature_permutation.attribute(inputs, target=labels)
    
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

def get_saliency_attributions(model, data_loader):
    saliency = Saliency(model)
    inputs, labels, bounding_boxes, _, _, _, _, _, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    attributions = saliency.attribute(inputs, target=labels)
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
    
    # IntegratedGradients
    parser.add_argument("--ig_n_steps", type=int, default=200, help="Number of steps for Integrated Gradients.")
    
    # DeepLift
    parser.add_argument("--dl_baseline", type=float, default=0.0, help="Baseline for DeepLift.")
    
    # DeepLiftShap
    parser.add_argument("--dls_baseline", type=float, default=0.0, help="Baseline for DeepLiftShap.")
    
    # GradientShap
    parser.add_argument("--gs_n_samples", type=int, default=20, help="Number of random samples per input for GradientShap.")
    parser.add_argument("--gs_baselines", type=float, default=0.0, help="Randomly sampled reference inputs for GradientShap.")
    
    # GuidedGradCam
    parser.add_argument("--ggc_layer", type=str, default="layer4[-1]", help="Specific layer for GuidedGradCAM.")
    
    # Occlusion
    parser.add_argument("--occ_sliding_window_shapes", type=int, nargs=3, default=[3, 15, 15], help="Size of the sliding window for Occlusion.")
    parser.add_argument("--occ_strides", type=int, nargs=3, default=[3, 8, 8], help="Stride values for the sliding window for Occlusion.")
    
    # ShapleyValueSampling
    parser.add_argument("--shapley_n_samples", type=int, default=25, help="Number of samples for Shapley Value Sampling.")
    
    # Lime
    parser.add_argument("--lime_n_samples", type=int, default=125, help="Number of samples for Lime.")
    parser.add_argument("--lime_ppe", type=int, default=10, help="Perturbations per evaluation for Lime.")
    
    # KernelShap
    parser.add_argument("--kernelshap_n_samples", type=int, default=200, help="Number of samples for KernelShap.")
    parser.add_argument("--kernelshap_baselines", type=float, default=0.0, help="Baseline samples for KernelShap.")
    
    # LRP
    parser.add_argument("--lrp_rule", type=str, default="Z", choices=["Z", "Z^+"], help="Propagation rule for LRP.")
    
    return parser.parse_args()

# =========================
# Visualisation
# =========================

def mask_to_bbox(mask):
    """
    Convert a binary mask to a bounding box.
    
    Args:
    - mask (torch.Tensor): A binary mask tensor of shape (H, W).
    
    Returns:
    - tuple: (x, y, w, h) coordinates of the bounding box.
    """
    print("Mask Shape:", mask.shape)

    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    y_min, y_max = torch.where(rows)[0][[0, -1]]
    x_min, x_max = torch.where(cols)[0][[0, -1]]

    return x_min.item(), y_min.item(), (x_max - x_min).item(), (y_max - y_min).item()


def save_attributions_bbox(inputs, attributions, bounding_boxes, save_dir, method_name):
    #print(bounding_boxes.shape)
    for i, input in enumerate(inputs):
        original_img = (input.cpu() - input.cpu().min()) / (input.cpu().max() - input.cpu().min())
        method_map = (attributions[i].cpu() - attributions[i].cpu().min()) / (attributions[i].cpu().max() - attributions[i].cpu().min())

        # Convert tensors to PIL images for drawing
        pil_original = transforms.ToPILImage()(original_img)
        pil_method_map = transforms.ToPILImage()(method_map)


        
        # Draw bounding box on the original image
        draw = ImageDraw.Draw(pil_original)
        #print(bounding_boxes[i].shape)
        bbox = bounding_boxes[i].tolist()  # convert tensor to list
        #print(bbox)
        width, height = pil_original.size


        
        # Convert bounding box from normalized to absolute coordinates
        x, y, w, h = mask_to_bbox(bounding_boxes[i].squeeze(0))
        if 0 <= x <= 1 and 0 <= y <= 1:  # This checks if the coordinates might be normalized
            x, y, w, h = x * width, y * height, w * width, h * height

        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=3)

        # Concatenate and save
        concatenated_images = torchvision.transforms.ToTensor()(torchvision.transforms.Resize((128, 128))(pil_original))
        concatenated_maps = torchvision.transforms.ToTensor()(torchvision.transforms.Resize((128, 128))(pil_method_map))
        final_images = make_grid([concatenated_images, concatenated_maps], nrow=2)
        save_image(final_images, os.path.join(save_dir, f"{method_name.lower()}_{i}.png"))




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
        "LRP": lambda: get_lrp_attributions(model, data_loader, rule=args.lrp_rule),
        "KernelShap": lambda: get_kernelshap_attributions(model, data_loader, n_samples=args.kernelshap_n_samples, baselines=args.kernelshap_baselines),
        "Lime": lambda: get_lime_attributions(model, data_loader, n_samples=args.lime_n_samples, perturbations_per_eval=args.lime_ppe),
        "ShapleyValueSampling": lambda: get_shapley_value_attributions(model, data_loader, n_samples=args.shapley_n_samples),
        "FeaturePermutation": get_feature_permutation_attributions,
        "Occlusion": lambda: get_occlusion_attributions(model, data_loader, sliding_window_shapes=args.occ_sliding_window_shapes, strides=args.occ_strides),
        "FeatureAblation": get_feature_ablation_attributions,
        "Deconvolution": get_deconvolution_attributions,
        "GuidedGradCam": lambda: get_guidedgradcam_attributions(model, data_loader, layer=args.ggc_layer),
        "GuidedBackprop": get_guidedbackprop_attributions,
        "InputXGradient": get_inputxgradient_attributions,
        "GradientShap": lambda: get_gradientshap_attributions(model, data_loader, n_samples=args.gs_n_samples, baselines=args.gs_baselines),
        "DeepLift": lambda: get_deeplift_attributions(model, data_loader, baseline=args.dl_baseline),
        "DeepLiftShap": lambda: get_deepliftshap_attributions(model, data_loader, baseline=args.dls_baseline),
        "Saliency": get_saliency_attributions,
        "IntegratedGradients": lambda: get_IntegratedGradients_attributions(model, data_loader, n_steps=args.ig_n_steps),
    }

    inputs, attributions, bounding_boxes = method_to_function[args.method]()
    
    # Calculate IoU for each image's attributions and bounding box
    k = 256  # or any value you desire
    ious = [calculate_iou(attribution.squeeze(0), bbox.squeeze(0), k) for attribution, bbox in zip(attributions, bounding_boxes)]
    
    # Optionally, print out the IoUs or store them for later analysis
    for idx, iou_val in enumerate(ious):
        print(f"Image {idx + 1}: IoU = {iou_val:.4f}")
    
    save_attributions_bbox(inputs, attributions, bounding_boxes, save_file_path, args.method)
    
