import os
import random
import h5py
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import sh
import torch
import torch as th
import torchvision
import wget
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, OneHotEncoder
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import torch.optim as optim
from torchvision import transforms, models
import wandb

# =========================
# Custom Dataset
# =========================

class SVHNDataset(Dataset):
    def __init__(
        self,
        file_path: str = "/home/htc/amahmud/ZIB_SummerIntern_Adnan",
        split: str = "train",
        target_digit: int = 1,
        transform: Optional[Callable] = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        target_transform: Optional[Callable] = None,
        balance: bool = True,
        max_samples: Optional[int] = None,
        one_vs_two: Optional[bool] = None,
        use_grayscale: Optional[bool] = None,
    ):
        """SVHN Dataset.

        Args:
            file_path (string): Path to the folder that contains the SVHN train, test or extra folders, e.g., ``.data/`.
            split (string): Split of the dataset. Needs to be "train" or "test".
            target_digit (int): Target digit to classify.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
            balance (bool, optional): Whether to balance the dataset or not.
            max_samples (int, optional): Maximum number of samples to use.
            one_vs_two (bool, optional): Whether to use the one vs two classification task or not.
            use_grayscale (bool, optional): Whether to use grayscale images or not.
        """
        self.split = split
        print("Loading SVHN data from", file_path, "for split", split, "...")
        print("Target digit:", target_digit)
        # Check if split is valid
        if self.split in ("train", "test", "extra"):
            file_path = os.path.join(file_path, f"{self.split}/digitStruct.mat")
        else:
            raise ValueError(f"split needs to be `train` or `test` or `extra`, got: {self.split}")

        # Load data in orignal format (hdf5)
        self.hdf5_data = h5py.File(file_path, "r")

        # Get target digit
        self.target_digit = target_digit
        self.transform = transform
        self.target_transform = target_transform
        self.balance = balance
        self.max_samples = max_samples
        self.one_vs_two = one_vs_two
        self.use_grayscale = use_grayscale

        assert self.target_digit in range(10), "target_digit needs to be in range(10)"

        # Get width and height of resizes from transforms
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchvision.transforms.Resize):
                    self.resize_height, self.resize_width = t.size  # type: ignore
                    break
        else:
            raise ValueError("transform cannot be None")

        # Convert hdf5 file to pandas dataframe
        self.df = self._convert_to_dataframe()
        if self.balance:
            self._balance_dataset()

        assert self.resize_width is not None and self.resize_height is not None

        # Create empty masks for merlin and morgana
        self.mask_merlin = torch.empty(
            (len(self.df), 1, self.resize_height, self.resize_width), dtype=torch.float32
        ).uniform_()
        self.mask_morgana = torch.empty(
            (len(self.df), 1, self.resize_height, self.resize_width), dtype=torch.float32
        ).uniform_()

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Converts the dataset to a pandas dataframe"""
        if self.split in ("train", "test", "extra"):
            print(
                f"Converting original {self.split} hdf5 file format to pandas dataframe... this might take a while ..."
            )
            print(
                "Alternatively, you can manually save the preprocessed dataframe to a csv file and load it from there."
            )
        # Get number of samples
        self.max_samples = self.max_samples if self.max_samples is not None else len(self.hdf5_data["/digitStruct/bbox"])  # type: ignore
        data = []
        for i in range(self.max_samples):
            img_name = self._get_name(i)
            bbox = self._get_bbox(i)
            data.append([img_name, bbox["label"], bbox["left"], bbox["top"], bbox["width"], bbox["height"]])

        # Create dataframe
        df = pd.DataFrame(data, columns=["img_name", "digits", "left", "top", "width", "height"])

        if self.one_vs_two is None:
            # Check if the target digit is in the label and add corresponding binary label for all samples
            df["binary_label"] = df["digits"].apply(lambda x: 1 if self.target_digit in x else 0)
        else:
            # Create y=1 if one is in x and two is not and y=0 if two is in x and one is not
            df["binary_label"] = df["digits"].apply(
                lambda x: 1
                if self.target_digit in x and 2 not in x
                else 0
                if 2 in x and self.target_digit not in x
                else 2
            )
            # Drop samples with no target digit
            df = df[df["binary_label"] != 2]

        return df

    def _balance_dataset(self) -> None:
        if self.split in ("train", "test", "extra"):
            print(f"Balancing {self.split} dataset by default...")
        # Balance the dataset with respect to binary label
        minority_class_count = min(self.df["binary_label"].value_counts())
        self.df = (
            self.df.groupby("binary_label")
            .apply(lambda x: x.sample(n=minority_class_count, random_state=42))
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.df)  # type: ignore

    def set_mask(self, new_mask, mode, idx) -> None:
        """Replaces mask with optimized mask"""
        if mode == "merlin":
            self.mask_merlin[idx] = new_mask
        elif mode == "morgana":
            self.mask_morgana[idx] = new_mask
        else:
            raise ValueError(f"mode needs to be merlin or morgana, got: {mode}")

    def __getitem__(self, index):
        # Get data from dataframe
        img_name = self.df.iloc[index]["img_name"]
        label = self.df.iloc[index]["binary_label"]
        bbox = {
            "digits": self.df.iloc[index]["digits"],
            "left": self.df.iloc[index]["left"],
            "top": self.df.iloc[index]["top"],
            "width": self.df.iloc[index]["width"],
            "height": self.df.iloc[index]["height"],
        }
        # Get image
        img = self._get_image(img_name)
        # Original image width and height
        W, H = img.size

        # Get the bounding box coordinates
        left = bbox["left"]
        top = bbox["top"]
        width = bbox["width"]
        height = bbox["height"]

        # Find the bounding box that contains all the bounding boxes and ensure that they are positive or zero
        x_left = max(0, min(left))
        y_top = max(0, min(top))
        x_right = min(left[-1] + width[-1], W)
        y_bottom = min(max(top) + max(height), H)

        # Choose a random cropping area that is within the original image and contains all the bounding boxes
        left_crop_coordinate = x_left - 10
        top_crop_coordinate = y_top - 10
        right_crop_coordinate = x_right + 10
        bottom_crop_coordinate = y_bottom + 10
        # 4-tuple defining the left, upper, right, and lower pixel
        cropping_area = (left_crop_coordinate, top_crop_coordinate, right_crop_coordinate, bottom_crop_coordinate)
        # Perform the cropping operation using the selected area
        img = img.crop(cropping_area)

        if self.transform:
            img = self.transform(img)

        if self.split == "test":
            # Update the bounding box coordinates to be relative to the cropped image
            left = [x - left_crop_coordinate for x in left]
            top = [y - top_crop_coordinate for y in top]

            # resized height and width due to transforms
            resized_height, resized_width = self.transform.transforms[0].size  # type: ignore

            # Update the bounding box coordinates to be scaled by the resize operation
            left = [int(round(x * resized_width / (right_crop_coordinate - left_crop_coordinate))) for x in left]
            top = [int(round(y * resized_height / (bottom_crop_coordinate - top_crop_coordinate))) for y in top]
            width = [int(round(w * resized_width / (right_crop_coordinate - left_crop_coordinate))) for w in width]
            height = [int(round(h * resized_height / (bottom_crop_coordinate - top_crop_coordinate))) for h in height]

            max_num_digit_ones = 3
            pad_value = -1

            # Filter the bounding box coordinates for the digit "1"
            one_indices = [i for i, digit in enumerate(bbox["digits"]) if digit == 1]
            left = [left[i] for i in one_indices]
            top = [top[i] for i in one_indices]
            width = [width[i] for i in one_indices]
            height = [height[i] for i in one_indices]

            # Pad the bounding box coordinate arrays with the pad_value
            one_bboxes_padded = list(zip(left, top, width, height)) + [(pad_value, pad_value, pad_value, pad_value)] * (
                max_num_digit_ones - len(left)
            )
            left, top, width, height = zip(*one_bboxes_padded)

            # plot the rectangles inside the image
            # fig, ax = plt.subplots(1)
            # ax.imshow(img.permute(1, 2, 0), vmin=img.min(), vmax=img.max())  # type: ignore
            # for i in range(len(left)):
            #     rect = patches.Rectangle(
            #         (left[i], top[i]), width[i], height[i], linewidth=1, edgecolor="r", facecolor="none"
            #     )
            #     ax.add_patch(rect)  # type: ignore
            # # save the plot
            # fig.savefig("plot.png")
            
            #print(f"Test Split - Label Type: {type(label)}, Label Value: {label}")
            
            return (
                img,
                label,
                self.mask_merlin[index],
                self.mask_morgana[index],
                torch.tensor(left),
                torch.tensor(top),
                torch.tensor(width),
                torch.tensor(height),
                index,
            )
        else:
            #print(f"Other Split - Label Type: {type(label)}, Label Value: {label}")
            return img, label, self.mask_merlin[index], self.mask_morgana[index], index

    def _get_name(self, index):
        """Get the image name for a given index, which is used to convert the dataset to a pandas dataframe"""
        name_ref = self.hdf5_data["/digitStruct/name"][index].item()  # type: ignore
        return "".join([chr(v[0]) for v in self.hdf5_data[name_ref]])  # type: ignore

    def _get_image(self, img_name):
        full_path = f"/home/htc/amahmud/ZIB_SummerIntern_Adnan/{self.split}/{img_name}"
        if self.use_grayscale is True:
            img = Image.open(full_path).convert("L")
        else:
            img = Image.open(full_path).convert("RGB")
        return img

    def _get_bbox(self, index):
        """Get the bounding box coordinates for a given index, which is used to convert the dataset to a pandas dataframe"""
        attrs = {}
        item_ref = self.hdf5_data["/digitStruct/bbox"][index].item()  # type: ignore
        for key in ["label", "left", "top", "width", "height"]:
            attr = self.hdf5_data[item_ref][key]  # type: ignore
            values = (
                [self.hdf5_data[attr[i].item()][0][0].astype(int) for i in range(len(attr))]  # type: ignore
                if len(attr) > 1  # type: ignore
                else [attr[0][0]]  # type: ignore
            )
            attrs[key] = values
        return attrs



# =========================
# CONFIGURATIONS
# =========================

# General configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50

# Dataset configurations
FILE_PATH = "/home/htc/amahmud/ZIB_SummerIntern_Adnan"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
TARGET_DIGIT = 1
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
TARGET_TRANSFORM = None
BALANCE = True
MAX_SAMPLES = None
ONE_VS_TWO = None
USE_GRAYSCALE = None
BATCH_SIZE = 32

# Model configurations
LEARNING_RATE = 0.001

CONFIG = {
    "DEVICE": DEVICE,
    "EPOCHS": EPOCHS,
    "FILE_PATH": FILE_PATH,
    "TRAIN_SPLIT": TRAIN_SPLIT,
    "TEST_SPLIT": TEST_SPLIT,
    "TARGET_DIGIT": TARGET_DIGIT,
    "TRANSFORM": str(TRANSFORM),  # transform to string as the actual object isn't serializable
    "BALANCE": BALANCE,
    "MAX_SAMPLES": MAX_SAMPLES,
    "ONE_VS_TWO": ONE_VS_TWO,
    "USE_GRAYSCALE": USE_GRAYSCALE,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
}

wandb.init(project="Merlin-Aurther Preliminary", config=CONFIG)

# =========================
# INIT W&B
# =========================

wandb.init(project="Merlin-Aurther Preliminary", config=locals())

# =========================
# DATA LOADING
# =========================

def load_dataset(split):
    dataset = SVHNDataset(
        file_path=FILE_PATH,
        split=split,
        target_digit=TARGET_DIGIT,
        transform=TRANSFORM,
        target_transform=TARGET_TRANSFORM,
        balance=BALANCE,
        max_samples=MAX_SAMPLES,
        one_vs_two=ONE_VS_TWO,
        use_grayscale=USE_GRAYSCALE
    )
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL SETUP
# =========================

def setup_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Modify for binary classification
    return model.to(DEVICE)

# =========================
# TRAINING LOOP
# =========================

MODEL_SAVE_PATH = "/home/htc/amahmud/ZIB_SummerIntern_Adnan/model"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        
        for images, labels, _, _, _ in train_loader:
            #print(f"Inside DataLoader loop - label type: {type(labels[0])}, label dtype: {labels[0].dtype if hasattr(labels[0], 'dtype') else 'N/A'}")
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Store test loss and accuracy for visualization
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Logging metrics to wandb
        wandb.log({
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        }, step=epoch)

        # Model checkpointing
        if (epoch + 1) % 10 == 0:  # +1 because epoch starts from 0
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    return train_losses, train_accuracies, test_losses, test_accuracies

# =========================
# EVALUATION LOOP
# =========================

def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, _, _, _, _, _, _, _ in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            
            total_loss += loss.item() * images.size(0)
            
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

# =========================
# VISUALIZATION
# =========================

def visualize_results(epochs, train_losses, train_accuracies, test_losses, test_accuracies):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plotting Training and Test Losses on ax1
    ax1.plot(epochs, train_losses, '-o', color='blue', label='Training Loss', markersize=4)
    ax1.plot(epochs, test_losses, '-o', color='cyan', label='Test Loss', markersize=4)
    
    ax1.set_title('Training & Test Loss and Accuracy across Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)  # Adding gridlines
    ax1.legend(loc='upper left')  # Adding a legend to ax1

    # Using twinx() to create a second y-axis for the accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracies, '-o', color='green', label='Training Accuracy', markersize=4)
    ax2.plot(epochs, test_accuracies, '-o', color='lime', label='Test Accuracy', markersize=4)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_yticks(np.arange(0, 1.1, 0.05))  # More refined y-ticks for accuracy
    ax2.legend(loc='upper right')  # Adding a legend to ax2

    plt.tight_layout()
    plt.savefig('/home/htc/amahmud/ZIB_SummerIntern_Adnan/Resnet-18_results.png')

    # Logging results to wandb
    for epoch, tr_loss, tr_acc, te_loss, te_acc in zip(epochs, train_losses, train_accuracies, test_losses, test_accuracies):
        wandb.log({
            "Epoch": epoch,
            "Train Loss": tr_loss,
            "Train Accuracy": tr_acc,
            "Test Loss": te_loss,
            "Test Accuracy": te_acc
        }, step=epoch)


# =========================
# DISPLAY HYPERPARAMETERS
# =========================

def display_hyperparameters():
    hyperparams_str = f"""
Hyperparameters:
BATCH_SIZE = {BATCH_SIZE}
EPOCHS = {EPOCHS}
LEARNING_RATE = {LEARNING_RATE}
DEVICE = {DEVICE}
FILE_PATH = {FILE_PATH}
TRAIN_SPLIT = {TRAIN_SPLIT}
TEST_SPLIT = {TEST_SPLIT}
TARGET_DIGIT = {TARGET_DIGIT}
BALANCE = {BALANCE}
MAX_SAMPLES = {MAX_SAMPLES}
ONE_VS_TWO = {ONE_VS_TWO}
USE_GRAYSCALE = {USE_GRAYSCALE}
"""
    print(hyperparams_str)

# =========================
# MAIN EXECUTION
# =========================

train_loader = load_dataset(TRAIN_SPLIT)
test_loader = load_dataset(TEST_SPLIT)
model = setup_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

display_hyperparameters()  # Display hyperparameters in the terminal

train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, train_loader, optimizer, criterion, EPOCHS)
visualize_results(range(1, EPOCHS+1), train_losses, train_accuracies, test_losses, test_accuracies)

# Save the model to wandb
torch.save(model.state_dict(), "model_weights.pth")
wandb.save("model_weights.pth")

# Visualize and log results
visualize_results(range(1, EPOCHS+1), train_losses, train_accuracies, test_losses, test_accuracies)

# Finish the wandb run
wandb.finish()
