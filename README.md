# Swin Transformer for Fish Species Classification

The goal of this project is to apply an image classification task using SOTA vision transformers model with the extra challenge of mitigating the issues related to the nature
of the chosen dataset (fish-vista) which happens to contains a lot of data: 60k+ images and large imbalances with minority classes that can go as low as 2 examples per class and
majority classes can be represented as high as 1500 on the training set.

[Link to the fish-vista Dataset](https://huggingface.co/datasets/imageomics/fish-vista)

## Dataset

The dataset consists of 69k images of fishes from 1758 unique species. Only a small subest of these species contains a sufficient number of training examples to build an accurate model.
I choose to keep only the species that have 50 examples or more in the training data which greatly reduces the number of classes down to 85.

<img width="988" height="523" alt="image" src="https://github.com/user-attachments/assets/a340beed-31f5-472b-bc63-39b628d0175f" />
