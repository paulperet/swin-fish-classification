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
    test_ultra_rare_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, bin='ultra_rare')
    test_ultra_rare_dataloader = DataLoader(test_ultra_rare_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_minority_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, bin='minority')
    test_minority_dataloader = DataLoader(test_minority_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_neutral_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, bin='neutral')
    test_neutral_dataloader = DataLoader(test_neutral_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_majority_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, bin='majority')
    test_majority_dataloader = DataLoader(test_majority_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    num_classes = len(test_ultra_rare_dataset.classes_to_idx)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model = load_model(test_ultra_rare_dataloader.dataset.classes_to_idx)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("--- Evaluating Model ---\n")

    ultra_rare_accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # Ultra Rare Bin
    with torch.no_grad():
        for images, labels in test_ultra_rare_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            ultra_rare_accuracy_metric.update(outputs, labels)

    ultra_rare_accuracy = ultra_rare_accuracy_metric.compute()
    ultra_rare_accuracy = ultra_rare_accuracy * 100
    print(f'Accuracy on ultra rare bin: {ultra_rare_accuracy:.2f}%')
    
    # Minority Bin
    minority_accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for images, labels in test_minority_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            minority_accuracy_metric.update(outputs, labels)
    
    minority_accuracy = minority_accuracy_metric.compute()
    minority_accuracy = minority_accuracy * 100
    print(f'Accuracy on minority bin: {minority_accuracy:.2f}%')

    # Neutral Bin
    neutral_accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for images, labels in test_neutral_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            neutral_accuracy_metric.update(outputs, labels)
    
    neutral_accuracy = neutral_accuracy_metric.compute()
    neutral_accuracy = neutral_accuracy * 100
    print(f'Accuracy on neutral bin: {neutral_accuracy:.2f}%')

    # Majority Bin
    majority_accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for images, labels in test_majority_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            majority_accuracy_metric.update(outputs, labels)
            
    majority_accuracy = majority_accuracy_metric.compute()
    majority_accuracy = majority_accuracy * 100
    print(f'Accuracy on majority bin: {majority_accuracy:.2f}%\n')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="./swin_model_checkpoint.pt",
        help="Path to the target model file. (.pt)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for data loading.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )