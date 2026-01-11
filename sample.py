from dataset import ImageFolderCustom
from torchvision import transforms
from transforms import train_transform, test_transform
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_classification = ImageFolderCustom("./data/classification_train.csv", transform=train_transform)
test_classification = ImageFolderCustom("./data/classification_test.csv", transform=test_transform)
val_classification = ImageFolderCustom("./data/classification_val.csv", transform=test_transform)

dataloader_train = DataLoader(train_classification, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test_classification, batch_size=32, shuffle=True)
dataloader_val = DataLoader(val_classification, batch_size=32, shuffle=True)

def plot_image(image, title="Image"):
    """Plot a single image with proper denormalization."""
    # Denormalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to numpy and transpose
    img_np = np.transpose(image.numpy(), (1, 2, 0))

    # Denormalize
    img_np = std * img_np + mean

    # Clip values to [0, 1] range
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Visualize a random image from the validation dataset
idx_to_label_train = {v: k for k, v in train_classification.classes_to_idx.items()}
i = np.random.randint(0, len(val_classification))
img, label = val_classification[i]
plot_image(img.squeeze(), title=f"Validation dataset example: {idx_to_label_train[label.item()]}")

# Visualize a random image from the train dataset
idx_to_label_train = {v: k for k, v in train_classification.classes_to_idx.items()}
i = np.random.randint(0, len(train_classification))
img, label = train_classification[i]
plot_image(img.squeeze(), title=f"Training dataset example: {idx_to_label_train[label.item()]}")