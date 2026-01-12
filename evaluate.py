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
    train_dataset = ImageFolderCustom("./data/classification_train.csv", transform=test_transform)
    val_dataset = ImageFolderCustom("./data/classification_val.csv", transform=test_transform)
    test_dataset = ImageFolderCustom("./data/classification_test.csv", transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model = load_model(train_dataloader.dataset.classes_to_idx)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("--- Evaluating Model ---\n")

    train_accuracy_metric = Accuracy(task="multiclass", num_classes=len(train_dataloader.dataset.classes_to_idx)).to(device)

    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            train_accuracy_metric.update(outputs, labels)

    train_accuracy = train_accuracy_metric.compute()
    train_accuracy = train_accuracy * 100
    print(f'Accuracy on train dataset: {train_accuracy:.2f}%')
    
    val_accuracy_metric = Accuracy(task="multiclass", num_classes=len(val_dataloader.dataset.classes_to_idx)).to(device)
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_accuracy_metric.update(outputs, labels)
    
    val_accuracy = val_accuracy_metric.compute()
    val_accuracy = val_accuracy * 100
    print(f'Accuracy on validation dataset: {val_accuracy:.2f}%')

    test_accuracy_metric = Accuracy(task="multiclass", num_classes=len(test_dataloader.dataset.classes_to_idx)).to(device)
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            test_accuracy_metric.update(outputs, labels)

    test_accuracy = test_accuracy_metric.compute()
    test_accuracy = test_accuracy * 100
    print(f'Accuracy on test dataset: {test_accuracy:.2f}%\n')

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