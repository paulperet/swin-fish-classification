# Swin Transformer for Fish Species Classification

The goal of this project is to apply an image classification task using a state-of-the-art vision transformer model with the extra challenge of mitigating the issues related to the nature
of the chosen dataset (fish-vista) which happens to contains a lot of data: 60k+ images and large imbalances with minority classes that can go as low as 2 examples per class and
majority classes can be represented as high as 1500 on the training set.

[Link to the fish-vista Dataset](https://huggingface.co/datasets/imageomics/fish-vista)

You can find the trained weights here :

[Link to the huggingface model](https://huggingface.co/Aelyos/swin-fish-classification)

## Dataset

### Description

The dataset consists of 69k images of fishes from 1758 unique species that are spread accross the training, validation and test set. Each of these dataset consists of a single csv file with a column corresponding to the fish species and another column corresponding to the name of the image file. There is also informations to build trait identification and segmentation models but we will not use it for this particular project.

Two randomly sampled images from the dataset:

<img width="1372" height="430" alt="image" src="https://github.com/user-attachments/assets/2e39ac5a-18c3-4f53-ae9a-27e5c14e0f8d" />

### Challenge
The dataset suffers from a severe class imbalance. As we can see around 10% of classes represent 80% samples. Only a small subset of these species contains a sufficient number of training examples to build an accurate model.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/8aff98ad-d0ac-4287-975b-e6ca20843128" />


## Mitigating class imbalance
The main goal of this project was to research and find techniques that would allow our vision transformer to reach competitive accuracy across all classes on a massive imbalanced dataset.

### Data Preprocessing & Augmentation
Following the recommendation of the original paper, I used a resizing technique that preserve the aspect ratio of the images so that the network can capture and discriminate each species on its proportions. It also come with the added benefit for our trained network to be used / fine tuned as a base model for another task such as trait segmentation, which is available with this dataset (Although the masks are only a subset of the entire fish dataset).

The augmentation used are random horizontal flips, random vertical flips, random rotations, color jittering and randomly adjusting the sharpness. They synthetically increase the number of unique examples, help the network avoid overfitting and assure that it truly learn each species features.

### Focal Loss
The issue with data that is imbalanced is that the network will learn on this distribution and pay more attention to the majority classes, in some case leading to a great overall accuracy that does not represent the true quality of our classifier.
There are two main ways to compensate for that:
- Using a weighted loss function
- Using a weighted random sampler

Following some research I have found that focal loss was addressing the imbalance issue. I ran two experiments, one with the weights for my classes are calculated as:

$$w_i = \frac{N}{n_i \cdot C}$$

Where $w_i$ corresponds to the weight for class $i$, $N$ is the total number of samples, $n_i$ is the number of samples in class $i$, and $C$ is the total number of unique classes

Then we apply log1p to smooth the range of the weights:

$$w_{i} = \ln\(w_{i})$$

Finally we normalize the weights:

$$\hat{w}_{i} = \frac{w_{i}}{\sum_{j=1}^{C} w_{j}} \times C$$

Where $w_i$ is the raw weight and $C$ is the number of unique species.

### Fine-Tuning a pretrained model
Starting from an already trained model as many benefits especially for computer vision. Indeed, we can expect that the pretrained network will have learned how to distinguish common features found in nature for example. Fine tuning will make the learning way faster, usually having a good accuracy from the first epochs and being better than random initialization [1].

## Training
I decided to compare two settings for focal loss. I used the normalized weights for alpha for my first model and left the default settings for the second. For the first 30 epochs I froze the model backbone and only updated the head weights. Then I unfroze all weights and train for 50 more epochs.

<img width="6000" height="1800" alt="image" src="https://github.com/user-attachments/assets/33d69a74-23a2-44a0-9f75-f829c4f11ee9" />


## Results

Here are the results for general accuracy and accuracy per bin (ultra rare 2-10, minority 10-100, neutral 100-500 and majority 500+). As we can observe the default focal loss without weights handle the class imbalance itself better than the weighted version. I made different experiments to scale the weights but it always failed to surpass the default focal loss. It is probably due to the range of weight values which is too large.

<img width="3200" height="960" alt="image" src="https://github.com/user-attachments/assets/f0429a76-6799-4d85-8245-5f89a24f9972" />


## References
[1] "All pre-training methods notably outperform random initialization. However, we
observe a considerably larger improvement under class imbalanced scenarios, where models pretrained on larger datasets yield greater boosts in accuracy." Ravid Shwartz-Ziv, Micah Goldblum, Yucen Lily Li, C. Bayan Bruss, & Andrew Gordon Wilson. (2023). Simplifying Neural Network Training Under Class Imbalance.
