# Swin Transformer for Fish Species Classification

The goal of this project is to apply an image classification task using a state-of-the-art vision transformer model with the extra challenge of mitigating the issues related to the nature
of the chosen dataset (fish-vista) which happens to contains a lot of data: 60k+ images and large imbalances with minority classes that can go as low as 2 examples per class and
majority classes can be represented as high as 1500 on the training set.

[Link to the fish-vista Dataset](https://huggingface.co/datasets/imageomics/fish-vista)

## Dataset

### Description

The dataset consists of 69k images of fishes from 1758 unique species that are spread accross the training, validation and test set. Each of these dataset consists of a single csv file with a column corresponding to the fish species and another column corresponding to the name of the image file. There is also informations to build trait identification and segmentation models but we will not use it for this particular project.

Two randomly sampled images from the dataset:

<img width="1372" height="430" alt="image" src="https://github.com/user-attachments/assets/2e39ac5a-18c3-4f53-ae9a-27e5c14e0f8d" />

### Challenge
The dataset suffers from a severe class imbalance. We can visualize it using the Lorenz curve (Gini values close to 0 signals equality while values close to 1 signals unequality). Only a small subset of these species contains a sufficient number of training examples to build an accurate model.
I choose to keep only the species that have 50 examples or more in the training data which greatly reduces the number of classes down to 85.

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/1b3a5165-04bc-46ba-9493-d6a9e28284aa" />

## Mitigating class imbalance
The main goal of this project was to research and find techniques that would allow our vision transformer to reach competitive accuracy across all classes on a massive imbalanced dataset.

### Data Preprocessing & Augmentation
Following the recommendation of the original paper, I used a resizing technique that preserve the aspect ratio of the images so that the network can capture and discriminate each species on its proportions. It also come with the added benefit for our trained network to be used / fine tuned as a base model for another task such as trait segmentation, which is available with this dataset (Although the masks are only a subset of the entire fish dataset).

The augmentation used are random horizontal flips, random vertical flips, random rotations, color jittering and randomly adjusting the sharpness. They synthetically increase the number of unique examples, help the network avoid overfitting and assure that it truly learn each species features.

### Weighted Cross Entropy Loss
The issue with data that is imbalanced is that the network will learn on this distribution and pay more attention to the majority classes, in some case leading to a great overall accuracy that does not represent the true quality of our classifier.
There are two main ways to compensate for that:
- Using a weighted loss function
- Using a weighted random sampler

I have decided to go with the weighted cross-entropy loss for this project. The weights for my classes are calculated as:

$$w_i = \frac{N}{n_i \cdot C}$$

Where $w_i$ corresponds to the weight for class $i$, $N$ is the total number of samples, $n_i$ is the number of samples in class $i$, and $C$ is the total number of unique classes

### Fine-Tuning a pretrained model
Starting from an already trained model as many benefits especially for computer vision. Indeed, we can expect that the pretrained network will have learned how to distinguish common features found in nature for example. Fine tuning will make the learning way faster, usually having a good accuracy from the first epochs and being better than random initialization [1].

[1] "All pre-training methods notably outperform random initialization. However, we
observe a considerably larger improvement under class imbalanced scenarios, where models pretrained on larger datasets yield greater boosts in accuracy." Ravid Shwartz-Ziv, Micah Goldblum, Yucen Lily Li, C. Bayan Bruss, & Andrew Gordon Wilson. (2023). Simplifying Neural Network Training Under Class Imbalance.
