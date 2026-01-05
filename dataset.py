import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image

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
    def __init__(self, targ_dir: str, transform=None, cutoff=50):
        
        # Get classes >50 in train
        train_most_represented = pd.read_csv('./data/classification_train.csv')
        counts = train_most_represented['standardized_species'].value_counts()
        classes_most_represented = train_most_represented[train_most_represented['standardized_species'].map(counts) >= cutoff]['standardized_species'].unique()

        # Filter from these classes
        df = pd.read_csv(targ_dir)
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

    def __len__(self):
        "Returns the total number of samples."
        return len(self.images)

    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.classes[index]
        class_idx = self.classes_to_idx[class_name]

        if self.transform:
            images = self.transform(img)
        return images , torch.tensor(class_idx, dtype=torch.long)
