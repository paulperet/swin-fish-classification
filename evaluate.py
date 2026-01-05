import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from dataset import ImageFolderCustom
from model import load_model
from transforms import test_transform

def evaluate(checkpoint_path, batch_size=256, num_workers=4):
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # Create test dataset & dataloader
    dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, cutoff=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = load_model(dataloader.dataset.classes_to_idx)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    accuracy_metric = Accuracy().to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            accuracy_metric.update(outputs, labels)

    accuracy = accuracy_metric.compute()
    print(f'Accuracy on test dataset: {accuracy:.2f}%')