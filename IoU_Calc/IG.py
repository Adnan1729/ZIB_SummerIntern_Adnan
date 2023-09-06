import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
from PIL import Image, ImageDraw
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from custom_dataset import SVHNDataset

# Constants
file_path = "/home/htc/amahmud/ZIB_SummerIntern_Adnan"
model_path = os.path.join(file_path, "model/model_epoch_50.pth")
save_file_path = os.path.join(file_path, "Quantus")
num_images_to_process = 5

# Ensure the save path exists
os.makedirs(save_file_path, exist_ok=True)

def load_model_and_dataset():
    # Load the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the dataset
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = SVHNDataset(file_path=file_path, split="test", target_digit=1, transform=transform)
    return model, dataset, device

def create_heatmap(attributions):
    attributions = np.mean(attributions, axis=0)
    norm_attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))
    heatmap = (plt.cm.jet(norm_attributions)[:, :, :3] * 255).astype(np.uint8)
    return np.transpose(heatmap, (2, 0, 1))


def process_image(input, label, left, top, width, height, pred_bbox=None):
    original_tensor_with_boxes = input.squeeze()
    if label.item() == 1:
        original_image_with_boxes = transforms.ToPILImage()(original_tensor_with_boxes.cpu().detach())
        draw = ImageDraw.Draw(original_image_with_boxes)
        
        # Draw ground truth bounding boxes in green
        for l, t, w, h in zip(left, top, width, height):
            draw.rectangle([l, t, l+w, t+h], outline="green", width=2)
        
        # If predicted bounding box is provided, draw it in red
        if pred_bbox:
            draw.rectangle([pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]], 
                           outline="red", width=2)
        
        original_tensor_with_boxes = transforms.ToTensor()(original_image_with_boxes)
    return original_tensor_with_boxes


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes. Format: [x_left, y_top, width, height].
    
    Returns:
    - iou: The IoU value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate union area
    union = w1 * h1 + w2 * h2 - intersection
    
    # Compute IoU
    iou = intersection / union if union != 0 else 0
    
    return iou

def heatmap_to_binary_mask(heatmap, threshold=0.8):
    """Converts a heatmap to a binary mask where values above the threshold are set to 1 and the rest are set to 0."""
    binary_mask = (heatmap >= threshold).float()
    return binary_mask


def extract_bbox_from_binary_mask(binary_mask):
    """Extracts a bounding box from a binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_top, y_bottom = np.where(rows)[0][[0, -1]]
    x_left, x_right = np.where(cols)[0][[0, -1]]
    
    return [x_left, y_top, x_right - x_left, y_bottom - y_top]


def compute_and_save_integrated_gradients(model, dataset, device):
    integrated_gradients = IntegratedGradients(model)
    for i, (input, label, _, _, left, top, width, height, _) in enumerate(dataset):
        if i >= num_images_to_process:
            break
        input = input.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)
        attributions = integrated_gradients.attribute(input, target=label).squeeze().cpu().detach().numpy()

        heatmap = create_heatmap(attributions)
        heatmap_tensor = torch.Tensor(heatmap)
        
        # Create binary mask for the heatmap
        binary_heatmap_mask = heatmap_to_binary_mask(heatmap_tensor[0])  # Assuming the heatmap is on the first channel

        # checking how many pixels are activated to investigate why the predcited bounding box is encompassing the entirity of the image. 
        activated_pixels = binary_heatmap_mask.sum().item()
        print(f"Number of activated pixels for image {i}: {activated_pixels}")


        # IoU based on your definition (heatmap vs bounding box)
        binary_bbox_mask = torch.zeros_like(binary_heatmap_mask)
        for l, t, w, h in zip(left, top, width, height):
            binary_bbox_mask[t:t+h, l:l+w] = 1
        
        intersection_defined = (binary_heatmap_mask * binary_bbox_mask).sum()
        union_defined = binary_heatmap_mask.sum() + binary_bbox_mask.sum() - intersection_defined
        iou_defined = intersection_defined / union_defined if union_defined != 0 else 0
        print(f"Defined IoU for image {i}: {iou_defined.item()}")
        
        # Traditional IoU (predicted bounding box vs ground truth bounding box)
        pred_bbox = extract_bbox_from_binary_mask(binary_heatmap_mask.numpy())
        iou_traditional = compute_iou((left[0], top[0], width[0], height[0]), pred_bbox)  # Assuming a single bounding box in ground truth for simplicity
        print(f"Traditional IoU for image {i}: {iou_traditional}")

        original_tensor_with_boxes = process_image(input, label, left, top, width, height, pred_bbox=pred_bbox)
        combined_tensor = torch.cat((original_tensor_with_boxes, heatmap_tensor), dim=2)
        save_image(combined_tensor, os.path.join(save_file_path, f"image_{i}.png"))



if __name__ == "__main__":
    model, dataset, device = load_model_and_dataset()
    compute_and_save_integrated_gradients(model, dataset, device)
    print("Integrated Gradients computation and saving complete!")
