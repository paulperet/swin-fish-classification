import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

def load_model(number_classes):
    # Load the pre-trained Swin-T model
    swin_t_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    # Number of classes in your dataset
    num_classes = len(number_classes) 

    # Get the number of input features from the original head
    num_features = swin_t_model.head.in_features

    # Additional linear layer and dropout layer on top of the original head
    swin_t_model.head = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return swin_t_model