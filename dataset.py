import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def find_classes(df):
    # Get class name and sort them
    classes = sorted(df['standardized_species'].unique())
    # Give an id to each class
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    # Give a weight depending on the amount of samples for each class
    weights = {class_to_idx[k]: len(df) / (v * len(df['standardized_species'].unique())) for k, v in df['standardized_species'].value_counts().items()}

    return class_to_idx, weights

def idx_to_label(id, classes_to_idx):
    inv_classes_to_idx = {v: k for k, v in classes_to_idx.items()}
    return inv_classes_to_idx[id]

def prediction_to_idx(probabilities):
    best_prediction = torch.argmax(probabilities)
    return best_prediction.item()

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None, cutoff=50, upper_bound=None):
        
        # Get classes >cutoff in train
        train_most_represented = pd.read_csv('./data/classification_train.csv', low_memory=False)
        counts = train_most_represented['standardized_species'].value_counts()

        if upper_bound != None:
            classes_most_represented = train_most_represented[(train_most_represented['standardized_species'].map(counts) >= cutoff) & (train_most_represented['standardized_species'].map(counts) <= upper_bound)]['standardized_species'].unique()
        else:
            classes_most_represented = train_most_represented[train_most_represented['standardized_species'].map(counts) >= cutoff]['standardized_species'].unique()

        # Filter from these classes
        df = pd.read_csv(targ_dir, low_memory=False)
        df = df[df['standardized_species'].isin(classes_most_represented)]

        images, labels = df['filename'], df['standardized_species']
        self.transform = transform
        self.classes = labels.to_list()
        self.images = images.to_list()
        self.classes_to_idx = find_classes(df)[0]
        self.classes_weights = find_classes(df)[1]

    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = os.path.join(os.getcwd(), './data/Images/', self.images[index])
        return Image.open(image_path)

    def resize_and_pad(self, image, target_size=224):
        """Resize image preserving aspect ratio, then pad to target size."""
        # Get original dimensions
        w, h = image.size

        # Calculate scaling factor to fit within target_size while preserving aspect ratio
        scale = min(target_size / w, target_size / h)

        # New dimensions after scaling
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Calculate padding needed
        pad_w = target_size - new_w
        pad_h = target_size - new_h

        # Distribute padding evenly on both sides
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        # Pad with white because its the dataset default background color
        return transforms.functional.pad(image, (left, top, right, bottom), fill=255)

    def __len__(self):
        "Returns the total number of samples."
        return len(self.images)

    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        img = self.resize_and_pad(img, target_size=224)
        class_name  = self.classes[index]
        class_idx = self.classes_to_idx[class_name]

        if self.transform:
            images = self.transform(img)
        return images , torch.tensor(class_idx, dtype=torch.long)
