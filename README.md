# Swin Transformer for Fish Species Classification

The goal of this project is to apply an image classification task using a State-of-the-art vision transformer model with the extra challenge of mitigating the issues related to the nature
of the chosen dataset (fish-vista) which happens to contains a lot of data: 60k+ images and large imbalances with minority classes that can go as low as 2 examples per class and
majority classes can be represented as high as 1500 on the training set.

[Link to the fish-vista Dataset](https://huggingface.co/datasets/imageomics/fish-vista)

## Dataset

### Description

The dataset consists of 69k images of fishes from 1758 unique species that are spread accross the training, validation and test set. Each of these datasets consists of a single csv files with a column corresponding to the fish species and another column corresponding to the name of the image file. There is also informations to build trait identification and segmentation models but we will not use it for this particular project.

An example of two randomly sampled images from the dataset:

<img width="1372" height="430" alt="image" src="https://github.com/user-attachments/assets/2e39ac5a-18c3-4f53-ae9a-27e5c14e0f8d" />

### Issues
The dataset suffers from a severe class imbalance. We can visualize it using the Lorenz curve (Gini values close to 0 signals equality while values close to 1 signals unequality). Only a small subest of these species contains a sufficient number of training examples to build an accurate model.
I choose to keep only the species that have 50 examples or more in the training data which greatly reduces the number of classes down to 85.

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/1b3a5165-04bc-46ba-9493-d6a9e28284aa" />

