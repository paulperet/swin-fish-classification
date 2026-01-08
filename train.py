import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import argparse
from pathlib import Path

# Import custom modules
from model import load_model
from dataset import ImageFolderCustom
from transforms import train_transform, test_transform

def train(checkpoint_path=None, output_path="model.pt", epochs_head=50, epochs_backbone=50, batch_size=256, num_workers=4, cutoff=50):
    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Create datasets & dataloaders
    train_classification = ImageFolderCustom("./data/classification_train.csv", transform=train_transform, cutoff=cutoff)
    test_classification = ImageFolderCustom("./data/classification_test.csv", transform=test_transform, cutoff=cutoff)
    val_classification = ImageFolderCustom("./data/classification_val.csv", transform=test_transform, cutoff=cutoff)

    dataloader_train = DataLoader(train_classification, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    dataloader_test = DataLoader(test_classification, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    dataloader_val = DataLoader(val_classification, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Load model
    model = load_model(train_classification.classes_to_idx)
    model = model.to(device)

    # Freeze only the backbone (not the head)
    for name, param in model.named_parameters():
        if not name.startswith('head'):
            param.requires_grad = False

    # Data parallelism if multiple GPUs are available
    if nn.DataParallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Get number of classes
    num_classes = len(train_classification.classes_to_idx)

    # Define loss function
    class_weights = torch.tensor([i[1] for i in sorted(train_classification.classes_weights.items())], dtype=torch.float32, device=device)

    # Use the weighted loss in CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Set optimizer with mixed precision
    use_amp = True
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        current_epoch_head = checkpoint['epoch'] + 1
        current_epoch_backbone = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
    else:
        current_epoch_head = 0
        current_epoch_backbone = 0
        best_val_loss = float('inf')

    # Print training info
    print("--- Training Info ---")
    print(f"Device used: {device}")
    print(f"Cutoff for classes: {cutoff}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(dataloader_train.dataset)}\n")
    print("--- Starting Training HEAD ---\n")

    # 1. Training loop - Head
    for epoch in range(current_epoch_head, epochs_head):  # loop over the dataset multiple times

        # switch model to training mode
        model.train()

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # print statistics
            running_loss += loss.item()

        # Switch model to evaluation
        model.eval()

        val_total_loss = 0.0

        # Validation loss calculation
        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

        # Save checkpoint if validation accuracy improves
        if epoch == 0 or val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "val_loss": val_total_loss,
                "epochs_head": epochs_head,
                "epochs_backbone": epochs_backbone,
                "cutoff": cutoff
            }
            torch.save(checkpoint, output_path)


        print(f'Epoch: {epoch + 1} loss: {running_loss / len(dataloader_train):.3f} Validation Loss: {val_total_loss / len(dataloader_val):.3f}')
        running_loss = 0.0

    print('Finished Training HEAD\n')
    print("--- Starting Training FULL ---\n")

    torch.cuda.empty_cache()

    # 2. Training loop - Backbone
    # Unfreeze the backbone and norm layers

    for param in model.features.parameters():
        param.requires_grad = True

    for param in model.norm.parameters():
        param.requires_grad = True
    
    # Use different learning rates for different parts of the model to avoid catastrophic forgetting

    use_amp = True
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-6, 'weight_decay': 0.05},  # Very low LR for backbone
        {'params': model.norm.parameters(), 'lr': 5e-6, 'weight_decay': 0.05},      # Slightly higher for norm
        {'params': model.head.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}       # Highest LR for head
    ])

    # Warmup for first 2 epochs, then cosine annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=2)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs_backbone-2, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[2])

    # Add gradient clipping
    max_grad_norm = 1.0

    for epoch in range(current_epoch_backbone, epochs_backbone):  # loop over the dataset multiple times

        # switch model to training mode
        model.train()

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # print statistics
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Switch model to evaluation
        model.eval()

        val_total_loss = 0.0

        # Validation loss calculation
        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
        
        # Save checkpoint if validation accuracy improves
        if epoch == 0 or val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": val_total_loss / len(dataloader_val),
                "epochs_head": epochs_head,
                "epochs_backbone": epochs_backbone,
                "cutoff": cutoff
            }
            torch.save(checkpoint, output_path)

        print(f'Epoch: {epoch + 1} loss: {running_loss / len(dataloader_train):.3f} Validation Loss: {val_total_loss / len(dataloader_val):.3f}')
        running_loss = 0.0

    print('Finished Training FULL\n')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("swin_model.pt"),
        help="Path to the target model file. (.pt)",
    )

    parser.add_argument(
        "--epochs-head",
        type=int,
        required=True,
        help="Number of training epoch for the head.",
    )

    parser.add_argument(
        "--epochs-backbone",
        type=int,
        required=True,
        help="Number of training epoch for the backbone.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training. (default: 256)",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to load the checkpoint. (default: None)",
    )

    parser.add_argument(
        "--cutoff",
        type=int,
        default=50,
        help="Cutoff for class representation. (default: 50)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading. (default: 4)",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train(
        output_path=args.output_file,
        epochs_head=args.epochs_head,
        epochs_backbone=args.epochs_backbone,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        num_workers=args.num_workers,
    )