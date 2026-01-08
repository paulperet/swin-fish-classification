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
    
    cutoff = torch.load(checkpoint_path)["cutoff"]

    # Create test dataset & dataloader
    train_dataset = ImageFolderCustom("./data/classification_train.csv", transform=test_transform, cutoff=cutoff)
    val_dataset = ImageFolderCustom("./data/classification_val.csv", transform=test_transform, cutoff=cutoff)
    test_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, cutoff=cutoff)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = load_model(train_dataloader.dataset.classes_to_idx)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    train_accuracy_metric = Accuracy().to(device)

    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            train_accuracy_metric.update(outputs, labels)
    
    val_accuracy_metric = Accuracy().to(device)
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_accuracy_metric.update(outputs, labels)
    
    test_accuracy_metric = Accuracy().to(device)
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            test_accuracy_metric.update(outputs, labels)

    train_accuracy = train_accuracy_metric.compute()
    print(f'Accuracy on train dataset: {train_accuracy:.2f}%')
    val_accuracy = val_accuracy_metric.compute()
    print(f'Accuracy on validation dataset: {val_accuracy:.2f}%')
    test_accuracy = test_accuracy_metric.compute()
    print(f'Accuracy on test dataset: {test_accuracy:.2f}%')