import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch.optim as optim
import torch.nn as nn

# Import custom modules
from model import load_model
from dataset import ImageFolderCustom, find_classes, idx_to_label, prediction_to_idx
from transforms import train_transform, test_transform

if __name__ == "__main__":
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # Create datasets & dataloaders
    train_classification = ImageFolderCustom("./data/classification_train.csv", transform=train_transform)
    test_classification = ImageFolderCustom("./data/classification_test.csv", transform=test_transform)
    val_classification = ImageFolderCustom("./data/classification_val.csv", transform=test_transform)

    dataloader_train = DataLoader(train_classification, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    dataloader_test = DataLoader(test_classification, batch_size=512, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    dataloader_val = DataLoader(val_classification, batch_size=512, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    # Load model
    model = load_model(train_classification.classes_to_idx)
    model = model.to(device)

    # Define loss function
    class_weights = torch.tensor([i[1] for i in sorted(train_classification.classes_weights.items())], dtype=torch.float32, device=device)

    # Use the weighted loss in CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Set optimizer with mixed precision
    use_amp = True
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

    # Torchmetrics metric
    train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    best_val_acc = 0.0

    for epoch in range(30):  # loop over the dataset multiple times

        # switch model to training mode
        model.train()

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()

        # Switch model to evaluation
        model.eval()

        # Training accuracy calculation
        with torch.no_grad():
            for images, labels in dataloader_train:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                train_accuracy.update(preds, labels)

        # Validation accuracy calculation
        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_accuracy.update(preds, labels)

        # Compute total for each accuracy
        train_total_accuracy = train_accuracy.compute()
        val_total_accuracy = val_accuracy.compute()

        # Save checkpoint if validation accuracy improves
        if epoch == 0 or val_total_accuracy > best_val_acc:
            best_val_acc = val_total_accuracy
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "val_acc": val_total_accuracy
            }
            torch.save(checkpoint, "checkpoint_start_best.tar")


        print(f'Epoch: {epoch + 1} loss: {running_loss / len(dataloader_train):.3f} Training Acc: {train_total_accuracy} Validation Acc: {val_total_accuracy}')
        running_loss = 0.0
        train_accuracy.reset()
        val_accuracy.reset()

    print('Finished Training')